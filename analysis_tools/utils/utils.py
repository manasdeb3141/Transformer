# Common utility functions for the project

def user_confirmation(prompt : str) -> bool:

    user_input = input(f"{prompt} (y/n): ")

    if user_input.lower() in ["yes", "y"]:
        return True
    else:
        return False