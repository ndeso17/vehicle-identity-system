from .utils_bbox import calculate_iou_over_a, get_center_point, is_point_inside_box

def associate_vehicle_to_plate(vehicle_bboxes, plate_bbox):
    """
    Associate a license plate to the best matching vehicle.
    Logic: Plate should be largely INSIDE the vehicle box.
    """
    best_iou = 0
    best_vehicle_idx = -1

    # Per spec: IoU = intersection / area(boxA). Use boxA = plate_bbox here
    for i, vehicle_bbox in enumerate(vehicle_bboxes):
        iou = calculate_iou_over_a(plate_bbox, vehicle_bbox)
        # Threshold > 0.05 per spec
        if iou > 0.05 and iou > best_iou:
            best_iou = iou
            best_vehicle_idx = i

    return best_vehicle_idx

def associate_driver_to_vehicle(vehicle_bbox, vehicle_type, person_bboxes):
    """
    Find the driver for a vehicle among detected persons.
    vehicle_type: 'motorcycle' or 'car' (or others)
    """
    # Keep old behavior as a weak fallback: compute standard IoU with driver ROI
    best_iou = 0
    best_person_idx = -1

    vx1, vy1, vx2, vy2 = vehicle_bbox
    vw = vx2 - vx1
    vh = vy2 - vy1

    if vehicle_type == 'motorcycle':
        driver_roi = [vx1, int(vy1 - vh * 0.5), vx2, int(vy1 + vh * 0.8)]
    elif vehicle_type == 'car':
        driver_roi = [vx1, vy1, vx2, int(vy1 + vh * 0.6)]
    else:
        driver_roi = vehicle_bbox

    # Use calculate_iou_over_a with boxA = person to approximate overlap ratio
    from .utils_bbox import calculate_iou_over_a
    for i, person_bbox in enumerate(person_bboxes):
        iou = calculate_iou_over_a(person_bbox, driver_roi)
        if iou > 0.3 and iou > best_iou:
            best_iou = iou
            best_person_idx = i

    return best_person_idx
