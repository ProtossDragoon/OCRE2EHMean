def parse_data(line, split_char=', '):
    line = line.split(split_char)
    p0, p1, p2, p3 = (
        (int(line[0]), int(line[1])), 
        (int(line[2]), int(line[3])), 
        (int(line[4]), int(line[5])), 
        (int(line[6]), int(line[7]))
    )
    sentence = line[-1].strip('\n')
    left = min([p0[0], p1[0], p2[0], p3[0]])
    top = min([p0[1], p1[1], p2[1], p3[1]])
    left_top = (left, top)
    right = max([p0[0], p1[0], p2[0], p3[0]])
    bottom = max([p0[1], p1[1], p2[1], p3[1]])
    right_bottom = (right, bottom)
    return {
        'polygon': [p0, p1, p2, p3], 
        'left_top': left_top,
        'right_bottom': right_bottom,
        'sentence': sentence,
    }


def parse_data_extended(line, split_char=', '):
    line = line.split(split_char)
    p0, p1, p2, p3 = (
        (int(line[0]), int(line[1])), 
        (int(line[2]), int(line[3])), 
        (int(line[4]), int(line[5])), 
        (int(line[6]), int(line[7]))
    )
    left_top = (int(line[8]), int(line[9]))
    right_bottom = (int(line[10]), int(line[11]))
    image_w, image_h = int(line[12]), int(line[13])
    sentence = line[-1].strip('\n')
    return {
        'polygon': [p0, p1, p2, p3],
        'left_top': left_top,
        'right_bottom': right_bottom,
        'image_w': image_w,
        'image_h': image_h,
        'sentence': sentence
    }