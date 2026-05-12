import unittest
import sys
import os
import json
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web.app import app
from src.database import init_db, get_db_connection

class TestSessionFlow(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Use a temporary DB or clear the existing one for strict testing
        # For this context, we'll just ensure tables exist
        init_db()

    def test_full_session_flow(self):
        # 1. Start Session
        res = self.app.post('/api/start_session')
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'started')
        session_id = data['session_id']
        self.assertIsNotNone(session_id)
        
        # Verify state via stats
        res = self.app.get('/api/stats')
        stats = json.loads(res.data)
        self.assertTrue(stats['session_active'])

        # 2. Simulate some time passing (logs should happen if app was running, 
        # but since we are in test client, the background loop in generate_frames isn't running.
        # We can manually trigger logging if we extracted it, but for now we test the API flow)
        
        # 3. End Session
        time.sleep(1) # Ensure start_time != end_time
        res = self.app.post('/api/end_session')
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'ended')

        # Verify state via stats
        res = self.app.get('/api/stats')
        stats = json.loads(res.data)
        self.assertFalse(stats['session_active'])

        # 4. Check persistence
        res = self.app.get('/api/sessions')
        sessions = json.loads(res.data)
        
        # Find our session
        my_session = next((s for s in sessions if s['id'] == session_id), None)
        self.assertIsNotNone(my_session)
        self.assertIsNotNone(my_session['end_time'])
        
        print(f"Verified Session {session_id}: Duration={my_session['end_time'] - my_session['start_time']}s")

if __name__ == '__main__':
    unittest.main()
