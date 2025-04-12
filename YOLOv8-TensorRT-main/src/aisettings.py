import json
class AISettings:
    def __init__(self, feed_config_file, ai_config_file) -> None:
        with open(feed_config_file, 'r') as file:
            self.config_data = json.load(file)
        #data = self.config_data[section_id]
        self.field_id = self.config_data.get('field', 0)
        self.duration_min = self.config_data.get('duration_min', 60)
        self.video_out_folder = self.config_data.get('video_out_folder', "")
        self.recording_id = self.config_data.get('recording_id', "123")
        self.sportunity_id = self.config_data.get('sportunity', "123")
        self.video_disabled = self.config_data.get('disable_video_generation', False)

        with open(ai_config_file, 'r') as file:
            self.ai_config_data = json.load(file)

        self.ball_confidence = self.ai_config_data.get('ball_confidence', 0.3)
        self.people_confidence = self.ai_config_data.get('people_confidence', 0.3)
        self.min_ball_size = self.ai_config_data.get('min_ball_size', 12)
        self.ball_mean_saturation = self.ai_config_data.get('ball_mean_saturation', 90)
        self.ball_mean_value = self.ai_config_data.get('ball_mean_value', 130)
        self.ball_do_deep_check = self.ai_config_data.get('ball_do_deep_check', True)

        self.detection_first_stage_frames = self.ai_config_data.get('detection_first_stage_frames', 6)

        self.detection_last_stage_on_best_frames = self.ai_config_data.get('detection_last_stage_on_best_frames', 10)
        self.detection_last_stage_on_both_centers_frames = self.ai_config_data.get('detection_last_stage_on_both_centers_frames', 8)