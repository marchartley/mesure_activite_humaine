"""! This script allows a Windows computer to access the server.
But it doesn't work for Linux
@TODO: adapt this code for Linux, if needed
"""

import os


accessed_servers = {}
def grant_access_server(hostname: str = None, share_folder_name: str = None, user: str = None, password: str = None,
						verbose: bool = False) -> bool:
	"""! Get an access between this computer and a remote server to be able to exchange files across the
	network
	@param hostname The IP address/URL of the server to grant access.
	By default in this project = lst-data2.local.univ-savoie.fr
	@param share_folder_name The shared folder you want to access.
	By default in this project = iot
	@param user Your username or at least the username of somebody that has access to the shared folder.
	@param password Your password
	@param verbose Display the results of this operation

	@return True if the operation is a success, meaning you now have an access corresponding on your official credential
	access. False otherwise.
	"""
	try:
		import pywintypes
		import win32wnet
		if user is None:
			user = 'mhartley'
		if password is None:
			password = 'achanger'
		if hostname is None:
			hostname = "lst-data2.local.univ-savoie.fr"
		if share_folder_name is None:
			share_folder_name = 'iot'
		SHARE_FULL_NAME = os.path.sep * 2 + os.path.sep.join((hostname, share_folder_name))

		if SHARE_FULL_NAME in accessed_servers and user in accessed_servers[SHARE_FULL_NAME]:
			if verbose:
				print(f"Already connected to {SHARE_FULL_NAME}")
			return True

		net_resource = win32wnet.NETRESOURCE()
		net_resource.lpRemoteName = SHARE_FULL_NAME
		# CONNECT_INTERACTIVE = 0x00000008
		flags = 0
		# flags |= CONNECT_INTERACTIVE
		if verbose:
			print("Trying to create connection to: {:s}".format(SHARE_FULL_NAME))
		try:
			win32wnet.WNetAddConnection2(net_resource, password, user, flags)
		except pywintypes.error as e:
			print(e)
			return False
		else:
			if verbose:
				print("Success!")
			if SHARE_FULL_NAME not in accessed_servers:
				accessed_servers[SHARE_FULL_NAME] = []
			if user not in accessed_servers[SHARE_FULL_NAME]:
				accessed_servers[SHARE_FULL_NAME].append(user)
			return True
	except ModuleNotFoundError as e:
		print('No access to the server granted, but that might be because you are not using a Linux OS (and don\'t '
			  'need it)')
		print(e)
		return False
