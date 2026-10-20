from datetime import datetime, date
import pytest
from db.connection import get_session, check_db_connection, init_db
from db.schema import Flight, DailyWeather, OptimizedSchedule


def test_init_db():
    backend = init_db()
    assert backend in ("mysql", "sqlite")
    conn_info = check_db_connection()
    assert conn_info["connected"] is True


def test_flight_crud():
    session = get_session()
    test_date = date(2026, 12, 1)

    # Clean existing
    session.query(Flight).filter(Flight.flight_number == "TEST999").delete()
    session.commit()

    flight = Flight(
        flight_number="TEST999",
        airline="Test Airlines",
        origin="JFK",
        destination="LAX",
        scheduled_departure=datetime(2026, 12, 1, 10, 0),
        scheduled_arrival=datetime(2026, 12, 1, 13, 0),
        departure_delay_min=12.5,
        status="On Time",
        flight_date=test_date,
    )
    session.add(flight)
    session.commit()

    saved = session.query(Flight).filter(Flight.flight_number == "TEST999").first()
    assert saved is not None
    assert saved.destination == "LAX"
    assert saved.departure_delay_min == 12.5

    # Dict representation
    f_dict = saved.to_dict()
    assert f_dict["flight_number"] == "TEST999"

    # Cleanup
    session.delete(saved)
    session.commit()
    session.close()


def test_daily_weather_crud():
    session = get_session()
    test_date = date(2026, 12, 2)

    session.query(DailyWeather).filter(DailyWeather.date == test_date).delete()
    session.commit()

    weather = DailyWeather(
        date=test_date,
        airport_code="JFK",
        temperature_c=22.5,
        wind_speed_kmh=18.0,
        precipitation_mm=5.0,
        visibility_km=8.5,
        weather_condition="Rain",
    )
    session.add(weather)
    session.commit()

    saved = session.query(DailyWeather).filter(DailyWeather.date == test_date).first()
    assert saved is not None
    assert saved.weather_condition == "Rain"
    assert saved.precipitation_mm == 5.0

    session.delete(saved)
    session.commit()
    session.close()


def test_optimized_schedule_crud():
    session = get_session()
    test_date = date(2026, 12, 3)

    session.query(OptimizedSchedule).filter(OptimizedSchedule.flight_number == "TESTOPT1").delete()
    session.commit()

    opt = OptimizedSchedule(
        flight_number="TESTOPT1",
        flight_date=test_date,
        original_departure=datetime(2026, 12, 3, 8, 0),
        predicted_delay_min=30.0,
        optimized_departure=datetime(2026, 12, 3, 8, 25),
        delay_adjustment_min=25.0,
        slot_or_runway="04L/22R",
        change_reason="Delayed by 25 mins due to high wind speed and runway capacity.",
    )
    session.add(opt)
    session.commit()

    saved = session.query(OptimizedSchedule).filter(OptimizedSchedule.flight_number == "TESTOPT1").first()
    assert saved is not None
    assert saved.delay_adjustment_min == 25.0
    assert "runway capacity" in saved.change_reason

    session.delete(saved)
    session.commit()
    session.close()
