from baduk_detect import detect_board_baduk, suggest_corners


def detect_board(img, board_size=19, threshold=0.035, corners=None):
    del threshold
    return detect_board_baduk(img, board_size=board_size, corners=corners, with_preview=False)
