# audio_feature_config.py
# Feature schema, range boundaries, and standard column names 

AUDIO_FEATURES = {
    'danceability': (0, 1),
    'energy': (0, 1),
    'key': (0, 11),
    'loudness': (-60, 0),
    'mode': (0, 1),
    'speechiness': (0, 1),
    'acousticness': (0, 1),
    'instrumentalness': (0, 1),
    'liveness': (0, 1),
    'valence': (0, 1),
    'tempo': (50, 200),
    'duration_ms': (10000, float('inf')),
    'time_signature': (1, 7)
}

NON_NORMALIZED_FEATURES = {'key', 'mode', 'time_signature', 'duration_ms', 'loudness'}

RENAME_MAP = {
    'Danceability': 'danceability',
    'Energy': 'energy',
    'Key': 'key',
    'Loudness': 'loudness',
    'Mode': 'mode',
    'Speechiness': 'speechiness',
    'Acousticness': 'acousticness',
    'Instrumentalness': 'instrumentalness',
    'Liveness': 'liveness',
    'Valence': 'valence',
    'Tempo': 'tempo',
    'Duration_ms': 'duration_ms',
    'duration': 'duration_ms',
    'Time_Signature': 'time_signature',
    'Time Signature': 'time_signature'
}
