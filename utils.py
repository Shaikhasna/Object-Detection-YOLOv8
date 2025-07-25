def get_valid_classes(user_input, all_classes):
    user_classes = [cls.strip().lower() for cls in user_input.split(",")]
    valid = [cls for cls in user_classes if cls in [x.lower() for x in all_classes]]
    return valid
