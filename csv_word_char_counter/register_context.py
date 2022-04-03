# This tool is used to register the app to run in a windows contextual menu.

import os
import sys
import winreg as reg


class RegisterContext:
    def __init__(self):
        # Find the path of main.py
        self.__app_path = os.path.dirname(os.path.abspath(__file__))
        self.__app_name = os.path.basename(self.__app_path)
        self.__app_key = r'Software\Classes\*\shell\{}'.format(self.__app_name)
        self.__app_command = '"{}" "%1"'.format(self.__app_path)
        # Context menu name
        self.__app_context_name = "Analyze CSV"

    # Checks if the app is already registered.
    @property
    def is_registered(self):
        try:
            key = reg.OpenKey(reg.HKEY_CLASSES_ROOT, self.__app_key)
            value = reg.QueryValue(key, '')
            if value == self.__app_name:
                return True
            else:
                return False
        except FileNotFoundError:
            return False

    # Registers the app to run in the windows contextual menu.
    # When the user right-clicks a csv file, the app can be run.
    def register(self):
        try:
            # Register first key with the context menu item name
            key = reg.CreateKey(reg.HKEY_CLASSES_ROOT, self.__app_key)
            reg.SetValue(key, '', reg.REG_SZ, self.__app_context_name)

            # Register second key with the command to run
            key_cmd = reg.CreateKey(reg.HKEY_CLASSES_ROOT, self.__app_key + r'\shell\open\command')
            reg.SetValue(key_cmd, '', reg.REG_SZ, self.__app_command)
        except Exception as e:
            print('Error: {}'.format(e))
            sys.exit(1)

    # Unregisters the app from the windows contextual menu.
    def unregister(self):
        try:
            reg.DeleteKey(reg.HKEY_CLASSES_ROOT, self.__app_key)
        except Exception as e:
            raise Exception('Error removing key: {}'.format(e))


