from collections import namedtuple
import time

Limits = namedtuple('Limits', ['min', 'max'])

bool_features = ('site', 'status', 'about', 'activities', 'interests', 'music', 'movies',
                 'tv', 'books', 'games', 'quotes', 'mobile_phone', 'home_phone', 'relation', 'has_photo', 'bdate',
                 'city', 'occupation', 'inspired_by', 'religion', 'domain')

VK_START_UNIX_TIME = 1254350400

model_limits = {
    **{x: Limits(0, 1) for x in bool_features},
    'sex': Limits(0, 2),
    'deactivated': Limits(0, 2),
    'last_seen': Limits(VK_START_UNIX_TIME, time.time())
}
