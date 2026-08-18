import json
with open('experiment_test/output/two_camera_28/predictions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        frame = data['frame_idx']
        if frame in [660, 1000, 2000, 3000, 3440]:
            b05 = data['cameras']['cam2']['parking_slots']['B05']
            o = b05['occupied']
            v = b05['vehicle_id']
            vo = b05['vision_occupied']
            to = b05['tracking_occupied']
            print('Frame', frame, 'Occupied:', o, 'VehicleID:', v, 'Vision:', vo, 'Tracking:', to)
