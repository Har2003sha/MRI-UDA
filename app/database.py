"""
Database layer.

Uses PostgreSQL when the DATABASE_URL environment variable is set
(e.g. postgresql://user:password@localhost:5432/mri_uda), and falls back to
a local SQLite file otherwise so the project runs out-of-the-box with zero
extra setup.

To use PostgreSQL:
    1. createdb mri_uda
    2. export DATABASE_URL=postgresql://user:password@localhost:5432/mri_uda
    3. pip install psycopg2-binary
    4. python app/app.py   (tables are created automatically on first run)
"""
import os
import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class SegmentationResult(db.Model):
    __tablename__ = "segmentation_results"

    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    scanner_domain = db.Column(db.String(50), nullable=False)   # "A" or "B"
    model_used = db.Column(db.String(50), nullable=False)       # "baseline" or "uda"

    input_image_path = db.Column(db.String(500), nullable=False)
    mask_image_path = db.Column(db.String(500), nullable=False)
    overlay_image_path = db.Column(db.String(500), nullable=False)

    tumor_pixel_count = db.Column(db.Integer, default=0)
    tumor_area_percent = db.Column(db.Float, default=0.0)

    # populated only when ground truth is available (synthetic demo samples)
    dice_score = db.Column(db.Float, nullable=True)
    iou_score = db.Column(db.Float, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "scanner_domain": self.scanner_domain,
            "model_used": self.model_used,
            "input_image_path": self.input_image_path,
            "mask_image_path": self.mask_image_path,
            "overlay_image_path": self.overlay_image_path,
            "tumor_pixel_count": self.tumor_pixel_count,
            "tumor_area_percent": round(self.tumor_area_percent, 2),
            "dice_score": round(self.dice_score, 3) if self.dice_score is not None else None,
            "iou_score": round(self.iou_score, 3) if self.iou_score is not None else None,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


def get_database_uri():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        # SQLAlchemy 1.4+/2.x requires 'postgresql://' not the legacy 'postgres://'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return db_url
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return f"sqlite:///{os.path.join(base_dir, 'mri_uda.db')}"
