"""
License Plate Bounding Box Refinement Module
=============================================

Modul ini berfungsi untuk memperbaiki (refine) bounding box plat nomor yang
dihasilkan oleh model YOLO agar lebih presisi dan benar-benar menempel pada
area plat nomor sebenarnya sebelum diproses OCR.

KONTEKS ILMIAH:
---------------
1. Mengapa contour-based refinement?
   - YOLO dirancang untuk generalisasi: bbox sering lebih besar dari objek
     untuk menghindari false negative, sehingga mengandung background noise
   - Contour-based approach memanfaatkan karakteristik fisik plat nomor:
     * High contrast edge (plat vs background)
     * Rectangular shape dengan aspect ratio konsisten (2:1 - 5:1)
     * Karakter yang terkelompok dalam satu region
   - Metode ini deterministik, explainable, dan tidak memerlukan model tambahan

2. Mengapa tidak langsung percaya bbox YOLO mentah?
   - Bbox yang terlalu besar akan memasukkan background noise ke OCR
   - OCR sangat sensitif terhadap preprocessing area yang tepat
   - Refinement meningkatkan precision tanpa mengurangi recall (fallback ada)
   - Studi menunjukkan bbox refinement dapat meningkatkan OCR accuracy 15-30%

3. Adaptasi untuk kondisi real-world:
   - CLAHE: mengatasi variasi pencahayaan (parkiran gelap vs terang siang)
   - Adaptive threshold: robust terhadap gradient pencahayaan
   - Morphological ops: menyatukan karakter yang terfragmentasi saat threshold
   - Fallback mechanism: jika gagal refine, tetap gunakan YOLO bbox

Target Environment:
------------------
- Python 3.10
- CPU only (Raspberry Pi 4)
- OpenCV + NumPy
- Lightweight & deterministic

Author: SPARXTrain CV Team
Date: 2025-12-26
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def refine_plate_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    confidence: float = 0.0,
    debug: bool = False,
    margin_shrink_pct: float = 0.04,
    min_area_ratio: float = 0.25
) -> Tuple[Tuple[int, int, int, int], Dict]:
    """
    Memperbaiki bounding box plat nomor menggunakan contour-based refinement.
    
    Algoritma:
    1. Crop ROI dari bbox YOLO awal
    2. Preprocessing: Grayscale → CLAHE → Gaussian Blur
    3. Adaptive Threshold untuk binarisasi robust
    4. Morphological operations: Closing (unite chars) → Opening (denoise)
    5. Cari contour terbesar dengan kriteria:
       - Aspect ratio mendekati plat (2:1 - 5:1)
       - Area minimal > 30% area ROI
    6. Map bbox baru ke koordinat global
    7. Fallback ke bbox YOLO jika refinement gagal
    
    Parameters:
    -----------
    image : np.ndarray
        Input image dalam format BGR (numpy array)
    bbox : Tuple[int, int, int, int]
        Bounding box awal dari YOLO (x, y, w, h) dalam pixel
    confidence : float, optional
        Confidence score dari YOLO detection (untuk logging)
    debug : bool, optional
        Jika True, return debug info lengkap untuk visualisasi
    margin_shrink_pct : float, optional
        Persentase margin shrink untuk menghilangkan edge pixels (default 0.04 = 4%)
        Nilai lebih besar = bbox lebih ketat, nilai 0 = no shrink
    min_area_ratio : float, optional
        Minimum ratio area contour terhadap ROI (default 0.25 = 25%)
    
    Returns:
    --------
    refined_bbox : Tuple[int, int, int, int]
        Refined bounding box (x, y, w, h) dalam pixel
    debug_info : Dict
        Dictionary berisi informasi debug:
        - 'original_bbox': bbox awal
        - 'refined_bbox': bbox hasil refinement
        - 'refinement_applied': boolean apakah refinement berhasil
        - 'refinement_method': "contour_snap" atau "fallback"
        - 'contour_area': area contour terpilih (jika ada)
        - 'aspect_ratio': aspect ratio contour terpilih (jika ada)
        - 'shrink_ratio': persentase shrinkage dari bbox awal
        - 'original_area': area bbox asli (pixels)
        - 'refined_area': area bbox refined (pixels)
        - 'roi_preprocessed': (jika debug=True) ROI setelah preprocessing
        - 'roi_binary': (jika debug=True) ROI setelah threshold
        - 'contours_drawn': (jika debug=True) visualisasi semua contour
    
    Example:
    --------
    >>> from plate_bbox_refinement import refine_plate_bbox
    >>> refined_bbox, debug = refine_plate_bbox(
    ...     image=frame,
    ...     bbox=plate_bbox,
    ...     confidence=0.45,
    ...     debug=True
    ... )
    >>> print(f"Shrink ratio: {debug['shrink_ratio']:.2%}")
    """
    
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    
    # Initialize debug info
    debug_info = {
        'original_bbox': bbox,
        'refined_bbox': bbox,
        'refinement_applied': False,
        'refinement_method': 'fallback',
        'contour_area': 0,
        'aspect_ratio': 0.0,
        'shrink_ratio': 0.0,
        'original_area': w * h,
        'refined_area': w * h,
        'confidence': confidence
    }
    
    # Validasi bbox tidak keluar dari frame
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        # Bbox invalid, return original
        return bbox, debug_info
    
    # === STEP 1: Crop ROI ===
    roi = image[y:y+h, x:x+w].copy()
    
    if roi.size == 0:
        return bbox, debug_info
    
    # === STEP 2: Preprocessing ===
    # Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Meningkatkan contrast lokal untuk mengatasi variasi pencahayaan
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian Blur ringan untuk mengurangi noise sebelum threshold
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    if debug:
        debug_info['roi_preprocessed'] = blurred.copy()
    
    # === STEP 3: Adaptive Threshold ===
    # Adaptive threshold lebih robust terhadap gradient pencahayaan
    # dibanding global threshold
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert: karakter jadi putih, background hitam
        blockSize=11,
        C=2
    )
    
    if debug:
        debug_info['roi_binary'] = binary.copy()
    
    # === STEP 4: Morphological Operations ===
    # Kernel untuk closing: menyatukan karakter yang berdekatan
    # Aspect ratio horizontal karena plat nomor berbentuk landscape
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Kernel untuk opening: menghilangkan noise kecil
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # === STEP 5: Cari Contour ===
    contours, _ = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if debug:
        # Visualisasi semua contour untuk debugging
        contours_vis = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contours_vis, contours, -1, (0, 255, 0), 1)
        debug_info['contours_drawn'] = contours_vis
    
    if len(contours) == 0:
        # Tidak ada contour ditemukan, fallback ke bbox awal
        return bbox, debug_info
    
    # Filter contour berdasarkan kriteria plat nomor
    roi_area = w * h
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Kriteria 1: Area minimal (default 25%) dari ROI
        # Lowered from 30% to allow tighter fits
        if area < min_area_ratio * roi_area:
            continue
        
        # Bounding rect dari contour
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        
        # Kriteria 2: Aspect ratio plat nomor (2:1 - 5:1 landscape)
        # Indonesia: umumnya 4:1 - 5:1
        # Toleransi lebar untuk berbagai jenis plat
        if ch == 0:
            continue
        
        aspect = cw / ch
        
        # Plat nomor Indonesia: landscape dengan aspect ratio 2.0 - 6.0
        # 6.0 untuk plat yang sangat panjang (truk)
        if not (2.0 <= aspect <= 6.0):
            continue
        
        valid_contours.append({
            'contour': cnt,
            'area': area,
            'bbox': (cx, cy, cw, ch),
            'aspect_ratio': aspect
        })
    
    if len(valid_contours) == 0:
        # Tidak ada contour valid, fallback
        return bbox, debug_info
    
    # Pilih contour dengan area terbesar
    best_contour = max(valid_contours, key=lambda c: c['area'])
    
    # === STEP 6: Map ke koordinat global ===
    cx, cy, cw, ch = best_contour['bbox']
    
    # Apply margin shrink for tighter fit (removes edge pixels)
    # This makes bbox stick precisely to plate edges
    if margin_shrink_pct > 0:
        shrink_px_x = int(cw * margin_shrink_pct)
        shrink_px_y = int(ch * margin_shrink_pct)
        
        # Ensure we don't shrink to nothing
        if shrink_px_x * 2 < cw and shrink_px_y * 2 < ch:
            cx += shrink_px_x
            cy += shrink_px_y
            cw -= 2 * shrink_px_x
            ch -= 2 * shrink_px_y
    
    # Konversi dari koordinat ROI ke koordinat global image
    refined_x = x + cx
    refined_y = y + cy
    refined_w = cw
    refined_h = ch
    
    # Validasi bbox baru tidak keluar dari frame
    refined_x = max(0, min(refined_x, img_w - 1))
    refined_y = max(0, min(refined_y, img_h - 1))
    refined_w = min(refined_w, img_w - refined_x)
    refined_h = min(refined_h, img_h - refined_y)
    
    refined_bbox = (refined_x, refined_y, refined_w, refined_h)
    
    # Calculate shrink ratio
    original_area = w * h
    refined_area = refined_w * refined_h
    shrink_ratio = 1.0 - (refined_area / original_area) if original_area > 0 else 0.0
    
    # Update debug info
    debug_info.update({
        'refined_bbox': refined_bbox,
        'refinement_applied': True,
        'refinement_method': 'contour_snap',
        'contour_area': best_contour['area'],
        'aspect_ratio': best_contour['aspect_ratio'],
        'shrink_ratio': shrink_ratio,
        'original_area': w * h,
        'refined_area': refined_w * refined_h
    })
    
    return refined_bbox, debug_info


def draw_refined_bbox(
    image: np.ndarray,
    original_bbox: Tuple[int, int, int, int],
    refined_bbox: Tuple[int, int, int, int],
    debug_info: Dict,
    show_original: bool = True
) -> np.ndarray:
    """
    Menggambar visualisasi debug bounding box refinement.
    
    Parameters:
    -----------
    image : np.ndarray
        Image untuk visualisasi (akan di-copy, tidak dimodifikasi)
    original_bbox : Tuple[int, int, int, int]
        Bounding box awal dari YOLO (x, y, w, h)
    refined_bbox : Tuple[int, int, int, int]
        Refined bounding box (x, y, w, h)
    debug_info : Dict
        Debug info dari refine_plate_bbox()
    show_original : bool, optional
        Jika True, tampilkan bbox original dengan warna berbeda
    
    Returns:
    --------
    vis_image : np.ndarray
        Image dengan visualisasi bbox
    
    Color Code:
    -----------
    - RED (0, 0, 255): Original YOLO bbox
    - GREEN (0, 255, 0): Refined bbox (sukses refinement)
    - YELLOW (0, 255, 255): Refined bbox (fallback, sama dengan original)
    """
    
    vis = image.copy()
    
    # Draw original bbox (RED)
    if show_original:
        x1, y1, w1, h1 = original_bbox
        cv2.rectangle(vis, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        cv2.putText(
            vis,
            "YOLO Original",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
    
    # Draw refined bbox
    x2, y2, w2, h2 = refined_bbox
    
    # Color: GREEN jika refinement applied, YELLOW jika fallback
    color = (0, 255, 0) if debug_info['refinement_applied'] else (0, 255, 255)
    label = "Refined" if debug_info['refinement_applied'] else "Fallback"
    
    cv2.rectangle(vis, (x2, y2), (x2+w2, y2+h2), color, 2)
    
    # Label dengan info detail
    if debug_info['refinement_applied']:
        info_text = f"{label} | AR:{debug_info['aspect_ratio']:.2f} | Shrink:{debug_info['shrink_ratio']:.1%}"
    else:
        info_text = f"{label} (No valid contour)"
    
    cv2.putText(
        vis,
        info_text,
        (x2, y2 + h2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )
    
    return vis


def visualize_refinement_process(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    debug_info: Dict
) -> np.ndarray:
    """
    Visualisasi lengkap proses refinement untuk debugging mendalam.
    
    Menampilkan:
    - ROI original
    - ROI setelah preprocessing (CLAHE + Blur)
    - ROI setelah binary threshold
    - Contours yang terdeteksi
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    bbox : Tuple[int, int, int, int]
        Original bounding box
    debug_info : Dict
        Debug info dari refine_plate_bbox() dengan debug=True
    
    Returns:
    --------
    combined : np.ndarray
        Image dengan 4 panel visualisasi proses
    """
    
    x, y, w, h = bbox
    
    # ROI original
    roi_original = image[y:y+h, x:x+w].copy()
    
    # Resize semua ke ukuran standar untuk visualisasi
    vis_height = 150
    vis_width = int(vis_height * (w / h)) if h > 0 else 300
    
    panels = []
    
    # Panel 1: Original ROI
    panel1 = cv2.resize(roi_original, (vis_width, vis_height))
    cv2.putText(panel1, "1. Original ROI", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    panels.append(panel1)
    
    # Panel 2: Preprocessed
    if 'roi_preprocessed' in debug_info:
        panel2 = cv2.resize(debug_info['roi_preprocessed'], (vis_width, vis_height))
        panel2_bgr = cv2.cvtColor(panel2, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2_bgr, "2. CLAHE + Blur", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(panel2_bgr)
    
    # Panel 3: Binary
    if 'roi_binary' in debug_info:
        panel3 = cv2.resize(debug_info['roi_binary'], (vis_width, vis_height))
        panel3_bgr = cv2.cvtColor(panel3, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel3_bgr, "3. Adaptive Threshold", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(panel3_bgr)
    
    # Panel 4: Contours
    if 'contours_drawn' in debug_info:
        panel4 = cv2.resize(debug_info['contours_drawn'], (vis_width, vis_height))
        cv2.putText(panel4, "4. Contours Detected", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(panel4)
    
    # Combine panels horizontally
    if len(panels) > 0:
        combined = np.hstack(panels)
    else:
        combined = roi_original
    
    return combined


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def batch_refine_plates(
    image: np.ndarray,
    detections: list,
    confidence_threshold: float = 0.3,
    debug: bool = False
) -> list:
    """
    Batch processing untuk refine multiple plate detections dalam satu image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    detections : list
        List of plate detections, format setiap item:
        {'bbox': (x, y, w, h), 'confidence': float}
    confidence_threshold : float
        Hanya refine detections dengan confidence >= threshold
    debug : bool
        Debug mode
    
    Returns:
    --------
    refined_detections : list
        List of refined detections dengan format yang sama + debug_info
    """
    
    refined = []
    
    for det in detections:
        bbox = det.get('bbox')
        conf = det.get('confidence', 0.0)
        
        if conf < confidence_threshold:
            # Skip low confidence detections
            continue
        
        refined_bbox, debug_info = refine_plate_bbox(
            image=image,
            bbox=bbox,
            confidence=conf,
            debug=debug
        )
        
        refined.append({
            'bbox': refined_bbox,
            'confidence': conf,
            'debug_info': debug_info
        })
    
    return refined


if __name__ == "__main__":
    """
    Contoh penggunaan dan testing module.
    Run: python plate_bbox_refinement.py
    """
    
    print("=" * 70)
    print("License Plate Bounding Box Refinement Module")
    print("=" * 70)
    print("\nModule ini dirancang untuk:")
    print("1. Memperbaiki bbox plat nomor dari YOLO detection")
    print("2. Meningkatkan precision untuk OCR preprocessing")
    print("3. Robust terhadap variasi pencahayaan (parkiran, jalan, CCTV)")
    print("\nContoh penggunaan:")
    print("-" * 70)
    print("""
from plate_bbox_refinement import refine_plate_bbox, draw_refined_bbox

# Single plate refinement
refined_bbox, debug = refine_plate_bbox(
    image=frame,
    bbox=(100, 200, 300, 80),  # YOLO bbox: x, y, w, h
    confidence=0.75,
    debug=True
)

# Visualisasi hasil
vis = draw_refined_bbox(
    image=frame,
    original_bbox=(100, 200, 300, 80),
    refined_bbox=refined_bbox,
    debug_info=debug,
    show_original=True
)

cv2.imshow("Refined Plate", vis)
cv2.waitKey(0)

# Debugging detail proses
process_vis = visualize_refinement_process(
    image=frame,
    bbox=(100, 200, 300, 80),
    debug_info=debug
)
cv2.imshow("Refinement Process", process_vis)
cv2.waitKey(0)
    """)
    print("=" * 70)
    print("\nModule ready untuk integrasi ke pipeline Flask!")
    print("Lokasi: App/Libs/plate_bbox_refinement.py")
    print("=" * 70)
