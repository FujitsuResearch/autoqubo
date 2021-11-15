# (major, minor, patch, prerelease)

VERSION = (0, 0, 1, "")
__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = 'autoqubo'
__contact_names__ = 'Fujitsu Research of Europe, Ltd.'
__contact_emails__ = ''
__homepage__ = ''
__repository_url__ = ''
__download_url__ = ''
__description__ = 'AutoQUBO gives you the tools for creating QUBO from Python code.'
__license__ = ''
__keywords__ = 'QUBO'
