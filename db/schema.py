from datetime import datetime, date, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Date,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Flight(Base):
    """Stores raw and ingested flight schedule data."""
    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    flight_number = Column(String(20), nullable=False, index=True)
    airline = Column(String(64), nullable=False)
    origin = Column(String(10), nullable=False)
    destination = Column(String(10), nullable=False)
    scheduled_departure = Column(DateTime, nullable=False)
    scheduled_arrival = Column(DateTime, nullable=False)
    actual_departure = Column(DateTime, nullable=True)
    actual_arrival = Column(DateTime, nullable=True)
    departure_delay_min = Column(Float, nullable=True, default=0.0)
    arrival_delay_min = Column(Float, nullable=True, default=0.0)
    aircraft_type = Column(String(32), nullable=True)
    assigned_runway = Column(String(20), nullable=True)
    status = Column(String(32), nullable=False, default="Scheduled")
    flight_date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to optimized schedule
    optimizations = relationship("OptimizedSchedule", back_populates="flight", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_flight_date_number", "flight_date", "flight_number"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "flight_number": self.flight_number,
            "airline": self.airline,
            "origin": self.origin,
            "destination": self.destination,
            "scheduled_departure": self.scheduled_departure.isoformat() if self.scheduled_departure else None,
            "scheduled_arrival": self.scheduled_arrival.isoformat() if self.scheduled_arrival else None,
            "actual_departure": self.actual_departure.isoformat() if self.actual_departure else None,
            "actual_arrival": self.actual_arrival.isoformat() if self.actual_arrival else None,
            "departure_delay_min": self.departure_delay_min,
            "arrival_delay_min": self.arrival_delay_min,
            "aircraft_type": self.aircraft_type,
            "assigned_runway": self.assigned_runway,
            "status": self.status,
            "flight_date": self.flight_date.isoformat() if self.flight_date else None,
        }


class DailyWeather(Base):
    """Stores airport weather observations and forecasts."""
    __tablename__ = "daily_weather"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    airport_code = Column(String(10), nullable=False, default="JFK")
    temperature_c = Column(Float, nullable=True)
    wind_speed_kmh = Column(Float, nullable=True)
    precipitation_mm = Column(Float, nullable=True)
    visibility_km = Column(Float, nullable=True)
    weather_condition = Column(String(64), nullable=True)  # e.g., Clear, Fog, Rain, Thunderstorm
    humidity_pct = Column(Float, nullable=True)
    wind_direction_deg = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "date": self.date.isoformat() if self.date else None,
            "airport_code": self.airport_code,
            "temperature_c": self.temperature_c,
            "wind_speed_kmh": self.wind_speed_kmh,
            "precipitation_mm": self.precipitation_mm,
            "visibility_km": self.visibility_km,
            "weather_condition": self.weather_condition,
            "humidity_pct": self.humidity_pct,
            "wind_direction_deg": self.wind_direction_deg,
        }


class OptimizedSchedule(Base):
    """Stores LLM-generated conflict-free, delay-mitigated flight timetable."""
    __tablename__ = "optimized_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    flight_id = Column(Integer, ForeignKey("flights.id", ondelete="CASCADE"), nullable=True)
    flight_number = Column(String(20), nullable=False)
    flight_date = Column(Date, nullable=False, index=True)
    original_departure = Column(DateTime, nullable=False)
    predicted_delay_min = Column(Float, nullable=False, default=0.0)
    optimized_departure = Column(DateTime, nullable=False)
    delay_adjustment_min = Column(Float, nullable=False, default=0.0)
    slot_or_runway = Column(String(30), nullable=True)
    change_reason = Column(Text, nullable=False)
    optimization_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Back reference
    flight = relationship("Flight", back_populates="optimizations")

    def to_dict(self):
        return {
            "id": self.id,
            "flight_id": self.flight_id,
            "flight_number": self.flight_number,
            "flight_date": self.flight_date.isoformat() if self.flight_date else None,
            "original_departure": self.original_departure.isoformat() if self.original_departure else None,
            "predicted_delay_min": self.predicted_delay_min,
            "optimized_departure": self.optimized_departure.isoformat() if self.optimized_departure else None,
            "delay_adjustment_min": self.delay_adjustment_min,
            "slot_or_runway": self.slot_or_runway,
            "change_reason": self.change_reason,
            "optimization_timestamp": self.optimization_timestamp.isoformat() if self.optimization_timestamp else None,
        }
