import quaternion


def quatdiff_in_euler(quat_curr, quat_des):

    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)

    rel_mat = des_mat.T.dot(curr_mat)

    rel_quat = quaternion.from_rotation_matrix(rel_mat)

    vec = quaternion.as_float_array(rel_quat)[1:]

    if rel_quat.w < 0.0:
        vec = -vec

    return -des_mat.dot(vec)

