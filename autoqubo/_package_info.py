# (major, minor, patch, prerelease)

VERSION = (0, 0, 3, "")
__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = 'autoqubo'
__contact_names__ = 'Fujitsu Limited'
__contact_emails__ = ''
__homepage__ = ''
__repository_url__ = 'https://github.com/FujitsuResearch/autoqubo'
__download_url__ = ''
__description__ = 'AutoQUBO gives you the tools for creating QUBO from Python code.'
__license__ = 'BSD-3-Clause'
__keywords__ = 'QUBO'
