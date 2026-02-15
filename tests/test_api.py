import base64
import unittest

import cv2
import numpy as np

from app import app, decode_image_data_url


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @staticmethod
    def make_frame_payload():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (180, 180), (255, 255, 255), -1)
        ok, encoded = cv2.imencode('.jpg', frame)
        assert ok
        return 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')

    def test_decode_invalid_data(self):
        self.assertIsNone(decode_image_data_url('not_base64'))

    def test_detect_invalid_mode(self):
        response = self.client.post('/api/detect/unknown', json={'frame': self.make_frame_payload()})
        self.assertEqual(response.status_code, 404)

    def test_detect_missing_frame(self):
        response = self.client.post('/api/detect/object', json={})
        self.assertEqual(response.status_code, 400)

    def test_detect_movement_success(self):
        response = self.client.post('/api/detect/movement', json={'client_id': 't1', 'frame': self.make_frame_payload()})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload.get('ok'))
        self.assertTrue(payload.get('image', '').startswith('data:image/jpeg;base64,'))
        self.assertIsInstance(payload.get('meta'), dict)

    def test_metrics_and_config_endpoints(self):
        metrics = self.client.get('/api/metrics')
        config = self.client.get('/api/config')
        health = self.client.get('/api/health')

        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(config.status_code, 200)
        self.assertEqual(health.status_code, 200)


if __name__ == '__main__':
    unittest.main()
