import sys
import mediapipe as mp
import inspect

print('python', sys.version)
print('mediapipe', mp.__version__)
print('has mp.solutions', hasattr(mp, 'solutions'))
if hasattr(mp, 'solutions'):
    print('has mp.solutions.hands', hasattr(mp.solutions, 'hands'))
    try:
        import mediapipe.solutions.hands as mh
        print('Hands class', mh.Hands)
        print('Hands sig', inspect.signature(mh.Hands))
    except Exception as e:
        print('inspect error', type(e).__name__, e)
else:
    print('mp.solutions not available')
