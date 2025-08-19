from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Date
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from typing import List, Optional
import os
import requests
import base64
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
import json

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:8081",  # Your frontend origin
    "http://localhost:3000",  # React frontend origin
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all request headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

DATABASE_URL = "mysql+mysqlconnector://admin:Hash#Alti#123@mhp-db.cbxzm6usxdiu.us-east-1.rds.amazonaws.com:3306/shipment"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# SQLAlchemy models matching given schema
class LoadType(Base):
    __tablename__ = "LoadType"

    loadTypeID = Column(Integer, primary_key=True, index=True)
    loadTypeName = Column(String(50), nullable=False)
    description = Column(String(255), nullable=True)

    orders = relationship("Order", back_populates="load_type")


class Order(Base):
    __tablename__ = "Orders"

    orderID = Column(Integer, primary_key=True, index=True)
    loadTypeID = Column(Integer, ForeignKey("LoadType.loadTypeID"), nullable=False)
    shipmentNumber = Column(String(100), nullable=False)
    pickupOrderNumber = Column(String(100), nullable=False)
    orderStatus = Column(String(50), nullable=True, default='Pending')
    origin = Column(Text, nullable=False)
    destination = Column(Text, nullable=False)
    bookingWeight = Column(Integer, nullable=False)
    loadCommodity = Column(String(255), nullable=False)
    packages = Column(Integer, nullable=False)
    warehouseName = Column(String(255), nullable=False)
    shipmentId = Column(String(100), nullable=False)
    pickupDate = Column(Date, nullable=False)  # Store as string if not using Date type
    pickupTime = Column(String(20), nullable=False)

    load_type = relationship("LoadType", back_populates="orders")


Base.metadata.create_all(bind=engine)


# Pydantic schemas for responses
class LoadTypeSchema(BaseModel):
    loadTypeID: int
    loadTypeName: str
    description: Optional[str] = None

    model_config = {
        "from_attributes": True
    }


class OrderSchema(BaseModel):
    orderID: int
    loadTypeID: int
    shipmentNumber: str
    pickupOrderNumber: str
    orderStatus: Optional[str] = 'Pending'
    origin: str
    destination: str
    bookingWeight: int
    loadCommodity: str
    packages: int
    warehouseName: str
    shipmentId: str
    pickupDate: date
    pickupTime: str

    model_config = {
        "from_attributes": True
    }


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/loadType", response_model=List[LoadTypeSchema])
def get_all_load_types(db: Session = Depends(get_db)):
    load_types = db.query(LoadType).all()
    return load_types


@app.get("/order/{loadTypeId}", response_model=List[OrderSchema])
def get_orders_by_load_type_id(loadTypeId: int, db: Session = Depends(get_db)):
    orders = db.query(Order).filter(Order.loadTypeID == loadTypeId).all()
    return orders


# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Use gpt-4o which has vision capabilities


@app.post("/analyzeImage")
async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        contents = await image.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        image_format = image.filename.split(".")[-1].lower()
        print("Model:", OPENAI_MODEL)
        print("API URL:", OPENAI_API_URL)
        
        # Only allow supported formats
        supported_formats = {"png", "jpeg", "jpg", "webp"}
        if image_format == "jpg":
            image_format = "jpeg"
        if image_format not in supported_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {image_format}. Supported formats: {', '.join(supported_formats)}")

        data_url_prefix = f"data:image/{image_format};base64,"
        data_url = data_url_prefix + base64_image

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an image analysis expert. When providing weight measurements, always use grams (g) as the unit. Convert any other weight units to grams."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        analysis = response.json()["choices"][0]["message"]["content"]

        return {"success": True, "analysis": analysis, "model": OPENAI_MODEL}

    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {http_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API to update order status
class UpdateOrderStatus(BaseModel):
    shipmentId: str

@app.put("/order/pickedup")
def update_order_status(request: UpdateOrderStatus, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.shipmentId == request.shipmentId).first()
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    order.orderStatus = "PickedUp"
    db.commit()
    db.refresh(order)
    return {"message": f"Order {request.shipmentId} status updated to PickedUp"}


# API to calculate metrics using AI
@app.post("/calculate-metrics")
async def calculate_metrics(
    image1: UploadFile = File(..., description="First image for analysis"),
    image2: UploadFile = File(..., description="Second image for analysis")
):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        # Read & encode images
        contents1 = await image1.read()
        base64_image1 = base64.b64encode(contents1).decode("utf-8")
        image_format1 = image1.filename.split(".")[-1].lower()

        contents2 = await image2.read()
        base64_image2 = base64.b64encode(contents2).decode("utf-8")
        image_format2 = image2.filename.split(".")[-1].lower()

        # Validate image formats
        supported_formats = {"png", "jpeg", "jpg", "webp"}
        for img_format, filename in [(image_format1, image1.filename), (image_format2, image2.filename)]:
            if img_format == "jpg":
                img_format = "jpeg"
            if img_format not in supported_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image format in {filename}: {img_format}. Supported formats: {', '.join(supported_formats)}"
                )

        # Convert to data URLs
        data_url1 = f"data:image/{image_format1};base64,{base64_image1}"
        data_url2 = f"data:image/{image_format2};base64,{base64_image2}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        # Enhanced prompt for volume calculation
        enhanced_prompt = """
        MEASURE THE OBJECT NOW. DO NOT SAY YOU CANNOT MEASURE.
        
        You are a measurement expert. You MUST provide actual measurements.
        
        CRITICAL: The credit card is 8.56cm × 5.398cm. Use this as your scale reference.
        
        REQUIREMENTS:
        - Length: Measure the longest side in cm
        - Breadth: Measure the middle side in cm
        - Height: Measure the shortest side in cm
        - Volume: Calculate in cm³
        
        FORBIDDEN RESPONSES:
        - "I'm unable to provide precise measurements"
        - "However, you can use the credit card as a reference"
        - Any explanation about how to measure
        
        YOU MUST RESPOND WITH ONLY:
        Length: [number]cm
        Breadth: [number]cm
        Height: [number]cm
        Volume: [number]cm³
        
        MEASURE NOW. NO EXCUSES.
        """

        # Use the same payload structure as analyzeImage
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a measurement expert. You MUST provide actual measurements from images. Never say you cannot measure. Always estimate based on visible references."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": enhanced_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url1
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url2
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        print("Model:", OPENAI_MODEL)
        print("API URL:", OPENAI_API_URL)

        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        analysis = response.json()["choices"][0]["message"]["content"]

        return {
            "success": True,
            "analysis": analysis,
            "model": OPENAI_MODEL,
            "images_processed": 2,
            "message": "Volume calculation completed successfully"
        }

    except requests.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {http_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
