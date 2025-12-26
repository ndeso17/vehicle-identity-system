# Vehicle Identity System

Sistem Identifikasi Kendaraan dan Pengguna Lahan Parkir Berbasis Fusi Fitur Citra Video.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Deskripsi

Sistem ini melakukan deteksi kendaraan menggunakan YOLOv8, OCR plat nomor, dan pengelompokan identitas kendaraan menggunakan pendekatan fusi fitur. Mirip dengan Google Photos, sistem mengelompokkan observasi kendaraan berdasarkan:

1. **Prioritas 1**: Plat nomor (OCR)
2. **Prioritas 2**: Wajah pengendara (Face Embedding)
3. **Prioritas 3**: Fitur visual (Tipe + Warna kendaraan)

---

## 🚀 Fitur Utama

- ✅ Deteksi kendaraan (car, motorcycle, bus, truck)
- ✅ OCR plat nomor Indonesia
- ✅ Deteksi wajah pengendara
- ✅ Pengelompokan identitas otomatis
- ✅ Admin dashboard dengan statistik
- ✅ Gallery view (Google Photos style)
- ✅ Merge & Split identitas manual
- ✅ Verifikasi identitas
- ✅ Multi-source input (Image, Video, Webcam, IP Camera)
- ✅ Session-based authentication
- ✅ RESTful API

---

## 📦 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/ndeso17/vehicle-identity-system.git
cd vehicle-identity-system/App
```

### 2. Buat Virtual Environment

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (untuk OCR plat nomor)

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer dari: https://github.com/UB-Mannheim/tesseract/wiki
```

### 5. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di:

- **Local**: http://127.0.0.1:5000
- **Network**: http://[IP-ADDRESS]:5000

---

## 📁 Struktur Folder

```
App/
├── app.py                      # Entry point Flask
├── requirements.txt            # Dependencies
├── Controllers/
│   └── api_controller.py       # Request handlers
├── Libs/
│   ├── auth.py                 # Authentication module
│   ├── config.py               # Configuration settings
│   ├── identity_manager.py     # Identity matching logic
│   ├── pipeline.py             # Detection pipeline
│   ├── plat_nomor.py           # Plate detector
│   ├── jenis_kendaraan.py      # Vehicle detector
│   ├── warna_kendaraan.py      # Color detector
│   └── pengemudi_kendaraan.py  # Driver attribution
├── Models/
│   ├── __init__.py
│   └── database.py             # SQLAlchemy models
├── Routes/
│   ├── api.py                  # Main API routes
│   ├── admin_routes.py         # Admin UI routes
│   ├── auth_routes.py          # Login/logout routes
│   └── vehicle_api.py          # Vehicle API endpoints
├── Views/
│   ├── base.html               # Admin template
│   ├── index.html              # Guest home
│   ├── result.html             # Detection result
│   ├── video.html              # Live stream
│   ├── auth/
│   │   └── login.html          # Login page
│   └── admin/
│       ├── dashboard.html      # Dashboard
│       ├── vehicles.html       # Vehicle list
│       ├── vehicle_detail.html # Vehicle detail
│       ├── observations.html   # Observations table
│       ├── gallery.html        # Gallery view
│       ├── gallery_detail.html # Gallery detail
│       ├── merge_split.html    # Merge & Split
│       └── settings.html       # Settings
├── static/
│   ├── crops/                  # Vehicle/plate crops
│   ├── frames/                 # Original frames
│   └── annotated/              # Annotated images
└── instance/
    └── vehicle_identity.db     # SQLite database
```

---

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT SOURCES                                  │
├─────────────┬─────────────┬─────────────┬─────────────────────────────────┤
│   Image     │   Video     │   Webcam    │        IP Camera              │
│   Upload    │   Upload    │   Stream    │        RTSP/HTTP              │
└──────┬──────┴──────┬──────┴──────┬──────┴─────────────┬───────────────────┘
       │             │             │                     │
       └─────────────┴─────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DETECTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Plate Detection (YOLOv8)                                            │
│     └─→ OCR Text Extraction (Pytesseract)                               │
│                                                                          │
│  2. Vehicle Detection (YOLOv8)                                          │
│     └─→ Color Detection (HSV Analysis)                                  │
│                                                                          │
│  3. Driver Detection (Person → Vehicle Attribution)                     │
│     └─→ Face Embedding (Optional)                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       IDENTITY MATCHING                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Priority 1: PLATE TEXT (OCR Confidence ≥ 70%)                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  IF plate_text matches existing identity                         │  │
│   │     → UPDATE existing identity                                   │  │
│   │  ELSE                                                            │  │
│   │     → CREATE new identity (method: plate)                        │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼ (if plate OCR failed)                    │
│   Priority 2: FACE EMBEDDING (Similarity ≥ 65%)                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  IF face_embedding matches existing identity                     │  │
│   │     → UPDATE existing identity                                   │  │
│   │  ELSE                                                            │  │
│   │     → CREATE new identity (method: face)                         │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼ (if no face available)                   │
│   Priority 3: VISUAL FEATURES (Type + Color + Time)                     │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Calculate weighted similarity score                             │  │
│   │  IF score ≥ threshold                                            │  │
│   │     → UPDATE existing identity                                   │  │
│   │  ELSE                                                            │  │
│   │     → CREATE new identity (method: visual)                       │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATABASE STORAGE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   VehicleIdentity                    VehicleObservation                  │
│   ┌────────────────────────┐        ┌────────────────────────┐          │
│   │ id                     │   1:N  │ id                     │          │
│   │ plate_text             │◄───────│ vehicle_id (FK)        │          │
│   │ face_embedding         │        │ timestamp              │          │
│   │ vehicle_type           │        │ source_type            │          │
│   │ vehicle_color          │        │ plate_text             │          │
│   │ identity_method        │        │ plate_confidence       │          │
│   │ detection_count        │        │ image_path             │          │
│   │ verified               │        │ annotated_image_path   │          │
│   │ first_seen             │        │ driver_detected        │          │
│   │ last_seen              │        │ ...                    │          │
│   └────────────────────────┘        └────────────────────────┘          │
│                                                                          │
│   AuditLog                                                               │
│   ┌────────────────────────┐                                            │
│   │ id                     │                                            │
│   │ action (verify/merge)  │                                            │
│   │ entity_type            │                                            │
│   │ entity_id              │                                            │
│   │ details (JSON)         │                                            │
│   │ timestamp              │                                            │
│   └────────────────────────┘                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADMIN DASHBOARD                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [Dashboard]  [Vehicles]  [Observations]  [Gallery]  [Merge]  [Settings]│
│                                                                          │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│   │    Statistics    │  │  Vehicle Cards   │  │   Photo Gallery  │      │
│   │   - Total        │  │  - Thumbnails    │  │   - Groups       │      │
│   │   - Verified     │  │  - Plate text    │  │   - Filters      │      │
│   │   - By method    │  │  - Actions       │  │   - Details      │      │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                          │
│   User Actions:                                                          │
│   ├── Verify identity  → Mark as confirmed                              │
│   ├── Edit plate text  → Manual OCR correction                          │
│   ├── Merge identities → Combine duplicates                             │
│   ├── Split identity   → Separate wrong groupings                       │
│   └── Delete identity  → Remove with audit log                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Authentication

### Default Credentials

| Username   | Password      | Role          |
| ---------- | ------------- | ------------- |
| `admin`    | `admin123`    | Administrator |
| `operator` | `operator123` | Operator      |

### Protected Routes

Semua route `/admin/*` memerlukan login. Akses tanpa login akan redirect ke `/login`.

---

## 📡 API Documentation

### Base URL

```
http://localhost:5000
```

---

### 🔑 Authentication

| Endpoint  | Method   | Description |
| --------- | -------- | ----------- |
| `/login`  | GET/POST | Login page  |
| `/logout` | GET      | Logout user |

---

### 📤 Upload & Detection

#### Upload Image

```http
POST /api/image
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description           |
| --------- | ---- | -------- | --------------------- |
| `image`   | File | Yes      | Image file (jpg, png) |

**Response:**

```html
Rendered result.html dengan annotated image dan JSON output
```

---

### 🚗 Vehicle Identities

#### List Identities

```http
GET /api/identities
```

| Parameter  | Type   | Default | Description                      |
| ---------- | ------ | ------- | -------------------------------- |
| `page`     | int    | 1       | Page number                      |
| `per_page` | int    | 20      | Items per page                   |
| `status`   | string | all     | `all`, `verified`, `unverified`  |
| `method`   | string | all     | `all`, `plate`, `face`, `visual` |
| `search`   | string | -       | Search by plate text             |

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "plate_text": "B 1234 XYZ",
      "plate_confidence": 0.85,
      "vehicle_type": "car",
      "vehicle_color": "white",
      "identity_method": "plate",
      "detection_count": 5,
      "verified": true,
      "first_seen": "2025-12-20T10:30:00",
      "last_seen": "2025-12-27T02:30:00"
    }
  ],
  "pagination": {
    "page": 1,
    "pages": 10,
    "total": 200
  }
}
```

#### Get Single Identity

```http
GET /api/identities/{id}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "id": 1,
    "plate_text": "B 1234 XYZ",
    "observations": [...]
  }
}
```

#### Verify Identity

```http
POST /api/identities/{id}/verify
```

**Response:**

```json
{
  "success": true,
  "message": "Identity verified"
}
```

#### Unverify Identity

```http
POST /api/identities/{id}/unverify
```

#### Update Plate Text

```http
PUT /api/identities/{id}/plate
Content-Type: application/json
```

**Body:**

```json
{
  "plate_text": "B 5678 ABC"
}
```

#### Delete Identity

```http
DELETE /api/identities/{id}
```

#### Merge Identities

```http
POST /api/identities/merge
Content-Type: application/json
```

**Body:**

```json
{
  "primary_id": 1,
  "secondary_ids": [2, 3, 4]
}
```

**Response:**

```json
{
  "success": true,
  "message": "Merged 3 identities into #1",
  "merged_count": 3
}
```

#### Split Identity

```http
POST /api/identities/split
Content-Type: application/json
```

**Body:**

```json
{
  "identity_id": 1,
  "observation_ids": [5, 6, 7]
}
```

**Response:**

```json
{
  "success": true,
  "new_identity_id": 10,
  "message": "Created new identity #10 with 3 observations"
}
```

---

### 👁️ Observations

#### List Observations

```http
GET /api/observations
```

| Parameter     | Type   | Default | Description                                |
| ------------- | ------ | ------- | ------------------------------------------ |
| `page`        | int    | 1       | Page number                                |
| `per_page`    | int    | 50      | Items per page                             |
| `identity_id` | int    | -       | Filter by identity                         |
| `source`      | string | all     | `all`, `image`, `video`, `webcam`, `ipcam` |

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "vehicle_id": 1,
      "timestamp": "2025-12-27T02:30:00",
      "source_type": "image",
      "plate_text": "B 1234 XYZ",
      "plate_confidence": 0.85,
      "ocr_success": true,
      "vehicle_type": "car",
      "vehicle_color": "white",
      "image_path": "static/crops/vehicle_xxx.jpg"
    }
  ]
}
```

#### Get Single Observation

```http
GET /api/observations/{id}
```

#### Delete Observation

```http
DELETE /api/observations/{id}
```

---

### 📊 Statistics

#### System Statistics

```http
GET /api/stats
```

**Response:**

```json
{
  "success": true,
  "data": {
    "total_identities": 150,
    "verified_identities": 45,
    "unverified_identities": 105,
    "total_observations": 1250,
    "plate_based": 100,
    "face_based": 30,
    "visual_based": 20
  }
}
```

---

### 📝 Audit Log

#### Get Audit Log

```http
GET /api/audit
```

| Parameter | Type | Default | Description       |
| --------- | ---- | ------- | ----------------- |
| `limit`   | int  | 50      | Number of entries |

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "action": "verify",
      "entity_type": "identity",
      "entity_id": 5,
      "details": { "verified_at": "2025-12-27T02:30:00" },
      "timestamp": "2025-12-27T02:30:00"
    }
  ]
}
```

---

### 📹 Streaming

#### Webcam Stream (MJPEG)

```http
GET /api/webcam
```

Returns: `multipart/x-mixed-replace` MJPEG stream

#### IP Camera Stream (MJPEG)

```http
GET /api/ipcam?url={rtsp_url}
```

| Parameter | Type   | Required | Description          |
| --------- | ------ | -------- | -------------------- |
| `url`     | string | Yes      | RTSP/HTTP camera URL |

---

## ⚙️ Configuration

Edit `Libs/config.py` untuk mengubah settings:

```python
# Identity Matching Thresholds
PLATE_PRIMARY_CONF = 0.7      # OCR confidence untuk primary identity
FACE_SIM_THRESHOLD = 0.65    # Face similarity threshold
CLUSTER_MATCH_THRESHOLD = 0.5 # Minimum score untuk match

# Feature Weights
WEIGHT_PLATE = 3.0
WEIGHT_FACE = 2.0
WEIGHT_TYPE = 0.5
WEIGHT_COLOR = 0.5
WEIGHT_TIME = 0.5

# Time Window
TIME_WINDOW_HOURS = 2        # Temporal proximity window

# Storage Paths
CROPS_FOLDER = 'static/crops'
FRAMES_FOLDER = 'static/frames'
ANNOTATED_FOLDER = 'static/annotated'

# Pagination
VEHICLE_UI_PER_PAGE = 20
OBSERVATIONS_PER_PAGE = 50
```

---

## 🛠️ Development

### Reset Database

```bash
rm instance/vehicle_identity.db
python app.py
```

### Add New User

Edit `Libs/auth.py`:

```python
ADMIN_USERS = {
    'admin': 'admin123',
    'operator': 'operator123',
    'newuser': 'newpassword'  # Add new user
}
```

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👥 Contributors

- Vehicle Identity System Team

---

## 📧 Contact

For questions or support, please open an issue on GitHub.
