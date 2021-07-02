import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import warnings
import itertools
from sklearn import metrics
import warnings
import matplotlib.patches as mpathes
import time

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)  # 精确表示小数
np.seterr(invalid='ignore')


def read_dataset(data_embedding):  # 用于s1,s2,s3,s4文件数据的读取
    cluster_label = list()

    pos = dict()

    node_dict = dict()
    data_size = data_embedding.shape[0]
    for i in range(data_size):
        cluster_label.append(i)
        node_dict[i] = len(node_dict)
        pos[i] = data_embedding[i].tolist()  # 用于可视化的字典

    # print("The size and shape of dataset are: ", data_size, data_embedding.shape)

    return cluster_label, pos, node_dict, data_size


def distance_neighbor(node1, node2, data_embedding):
    v1 = data_embedding[node1]
    v2 = data_embedding[node2]
    distance = np.linalg.norm(np.array(v1 - v2))
    return distance


def angel_calculation(xi_id, s, data_embedding):
    v0 = data_embedding[xi_id]
    indice1 = s[0]
    indice2 = s[1]

    v1 = data_embedding[indice1] - v0
    v2 = data_embedding[indice2] - v0

    # v1_length = np.square(np.linalg.norm(v1))
    # v2_length = np.square(np.linalg.norm(v2))

    v1_length = np.linalg.norm(v1)
    v2_length = np.linalg.norm(v2)

    beta1 = 1 / (v1_length + v2_length + 1)  # 角度系数 受距离影响可能太大了
    # beta2 = 1 - (v1_length + v2_length) / 2 / max_radius

    angle0 = np.dot(v1, v2) / (v1_length * v2_length)

    if angle0 > 1:
        angle0 = 1
    elif angle0 < -1:
        angle0 = -1
    """添加角度的系数： 夹角两条边的长度和"""
    angle0 = (1 - angle0) * beta1

    """不考虑边长对角度的影响"""
    # angle0 = 1 - angle0

    return angle0


def merge_sub_arng(t, inners_l):
    """合并 子集 的合并"""
    sc = []
    lc = [-1 for i in range(t)]
    m = 0
    for p in range(t):
        if lc[p] == -1:
            lc[p] = m  # 以自己为中心
            sc.append([p])
            m += 1

        for q in range(p + 1, t):
            if not (p == q) and lc[p] != lc[q]:
                # print('intersection')
                l1 = len(inners_l[p])
                l2 = len(inners_l[q])
                l3 = len(set(inners_l[p]).intersection(inners_l[q]))
                if l3 / l2 >= 0.5 or l3 / l1 >= 0.5:
                    if lc[q] == -1:
                        lc[q] = lc[p]
                        sc[lc[p]].append(q)
                    else:
                        sc[lc[p]].extend(sc[lc[q]])
                        tt = sc[lc[q]].copy()
                        sc[lc[q]] = []
                        for x in tt:  # update label for points in sc[lc[p]]
                            lc[x] = lc[p]
    inners_union = []
    for x in sc:
        temp_ = []
        for y in x:
            temp_.extend(inners_l[y])
        inners_union.append(list(set(temp_)))
    return inners_union


def os_tnn_calculate(cluster_label, data_embedding, point_vector, t, dist):
    """
    计算数据点xi的欧氏距离的tnn，但包括自己的tnn，所以这里的tnn实际为: (t+1)NN
    :param data_embedding:
    :param point_vector:
    :param k:
    :return:
    """
    if dist == 0:
        temp = np.linalg.norm(data_embedding - point_vector, axis=1, keepdims=True).reshape(1, -1)
        simi_list = temp[0]
        simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
        kns0 = [cluster_label[simi_sorted[i]] for i in range(t + 1)]
        kvs0 = [simi_list[simi_sorted[i]] for i in range(t + 1)]
    else:
        kns0, kvs0 = list(), list()
        temp = np.linalg.norm(data_embedding - point_vector, axis=1, keepdims=True).reshape(1, -1)
        simi_list = temp[0]
        simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
        for i in simi_sorted:
            if simi_list[i] <= dist:
                kns0.append(cluster_label[i])
                kvs0.append(simi_list[i])
            else:
                break  # 因为这里的simi_sorted是按照距离升序排列的，所以一旦大于dist，则后面的值都否决

    return kns0, kvs0


def direction_division(i, data_embedding, mid, ptnn, t):
    """
    根据xi与tnn(xi)的中心点mid的向量(mid-xi)，tnn(xi)超球体内点xj与xi向量 (xj-xi), 两个向量的内积大于0表示xj位于以xi为中心的mid的半球内，否则位于反半球内。
    :param i:
    :param data_embedding:
    :param mid:
    :param ptnn:
    :return:
    """
    direction_temp, right_fxtnn_temp = [], []  # 保存xi的每个近邻与xi的方向：1表示在tnn/k方向，-1为反方向
    vi = data_embedding[i]
    vec_im = mid[i] - vi  # xi指向tnn/k 中心: mid[i]的向量
    for p in range(t):
        vp = data_embedding[ptnn[i][p]]
        vec_ip = vp - vi
        if np.dot(vec_im, vec_ip) <= 0:
            direction_temp.append(0)
        else:
            direction_temp.append(1)
            right_fxtnn_temp.append(ptnn[i][p])
    return direction_temp, right_fxtnn_temp


def region_information(data_size, ptnn, mid, data_embedding, t):
    """
    This function is used to calculate the reverse nearest neighbors of xi's tnn and xi
    """
    rnns, direction_tnn, loc_rnn, right_fxtnn = [], [], [], []  # 每个数据点的反向近邻, reverse nearest neighbors, rnns 保存xi与tnn之间的rnn关系，逻辑值1或0
    for i in range(data_size):
        """计算数据点的inverse k nearest neighbors－－－－－－－－－－－－－－－－－－－－－－－－－－"""
        r_list, l_list = [], []
        for node in ptnn[i]:
            if i in ptnn[node]:
                r_list.append(1)
                l_list.append(ptnn[node].tolist().index(i))  # 如果
            else:
                r_list.append(0)
                l_list.append(-1)
        rnns.append(r_list)
        loc_rnn.append(l_list)
        """计算数据点xi和它的tnn中心点的欧氏距离－－－－－－－－－－－－－－－－－－－－－－－－－－－－－"""
        direction_temp, right_fxtnn_temp = direction_division(i, data_embedding, mid, ptnn, t)
        direction_tnn.append(direction_temp)
        right_fxtnn.append(right_fxtnn_temp)
    return rnns, direction_tnn, loc_rnn


def myknn(data_embedding, k, data_size):  # 计算节点之间的欧氏距离
    pknn, pknd = [], []
    cluster_label = [j for j in range(data_embedding.shape[0])]

    for i in range(data_size):
        temp = np.linalg.norm(data_embedding - data_embedding[i], axis=1, keepdims=True).reshape(1, -1)
        simi_list = temp[0]
        simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
        kns0 = [cluster_label[simi_sorted[j]] for j in range(k + 1)]
        kvs0 = [simi_list[simi_sorted[j]] for j in range(k + 1)]

        for m in range(k+1):
            if kns0[m] == i:
                kns0.pop(m)
                kvs0.pop(m)
                break
        pknn.append(kns0)
        pknd.append(kvs0)
    pknn = np.array(pknn)
    pknd = np.array(pknd)
    return pknn, pknd


def large_nn(data_embedding, large_k, data_size):
    large_knn, large_knd = myknn(data_embedding, large_k, data_size)

    return large_knn, large_knd


def preprocess(data_size, data_embedding, t, k, large_knn, large_knd):  # 根据欧氏距离计算数据点的tnn和基于tnn的中心点的tnn
    """"""
    mid, midtnn, midtnd, bias_mid = [], [], [], []  # 这里的k近邻包含了自身

    pknn = large_knn[:, :k]
    pknd = large_knd[:, :k]
    ptnn = pknn[:, :t]
    ptnd = pknd[:, :t]

    pttnn, pttnn_data = [], []  # 保存每个点的近邻的近邻，理论上是10×10个节点， 10×10个点的向量构成的局部空间， 用于后续算法
    for i in range(data_size):
        """计算数据点xi: tnn－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－"""
        tnn_vector = [data_embedding[p].tolist() for p in ptnn[i]]
        tnn_vector.append(data_embedding[i].tolist())

        temp_data = tnn_vector.copy()
        temp_node = ptnn[i].tolist()
        temp_node.append(i)

        tnn_vector = np.array(tnn_vector)

        """计算数据点xi的tnn的中心点和它的tnn， 以及同距离的近邻和近邻数量－－－－－－－－－－－－－"""
        mid_xi = np.mean(tnn_vector, axis=0)  # 先计算tnn的中心点，到后面再去除自身作为近邻
        mid.append(mid_xi.tolist())  # xi的tnn的中心点向量添加到mid列表
        t1 = np.linalg.norm(data_embedding[i] - mid_xi)  # xi与tnn中心点的偏移长度
        bias_mid.append(t1.tolist())

        """这里的embedding 太大，我们选择取xi的tnn(xi)的tnn(xi)也就是在k*k个点范围内取tnn(mid(xi))"""
        for p in ptnn[i]:
            for q in ptnn[p]:
                if not (q in temp_node):
                    temp_node.append(q)
                    temp_data.append(data_embedding[q].tolist())
        pttnn.append(temp_node)
        pttnn_data.append(temp_data)
        tns1, tds1 = os_tnn_calculate(np.array(temp_node), np.array(temp_data), mid_xi, t, 0)  # 计算 mid 的 tnn ＃＃ 注意：这里的 k-1,但是调用函数里 k+1
        midtnn.append(tns1)  # 数据集里不存在mid，所以这里取k
        midtnd.append(tds1)  # 数据集里不存在mid，所以这里取k

    diver = np.sum(ptnd, axis=1)
    rnns, direction_tnn, loc_rnn = region_information(data_size, ptnn, mid, data_embedding, t)

    rnn_c = np.sum(np.array(rnns), axis=1)

    return ptnn, ptnd, pknn, pknd, bias_mid, mid, midtnn, midtnd, rnns, rnn_c, diver, direction_tnn, pttnn, pttnn_data, loc_rnn


def anti_relative_neighborhodd_graph(data_size, data_embedding, ptnn, t, ptnd, bias_mid):
    """
    在每个点xi的tnn范围内，提出了反相对近邻图, 计算tnn的分组，并计算点的平均加权角度. 用于后面的离群性判定.
    :param data_size:
    :param data_embedding:
    :param ptnn:
    :param k:
    :param ptnd:
    :param bias_mid:
    :return:
    """
    diver_tnn_fine = []
    angle, angle_refine = [], []
    coo = []

    for i in range(data_size):
        """计算 xi 的tnn 的 RNG """
        xi_vector = data_embedding[i]
        inners_l = list()  # 临时记录数据点的散度点列表
        radius_max = ptnd[i][-1]  # 所有的内部圆取相同的圆半径
        coo_1 = list()

        matrix_tnn = np.array([data_embedding[ptnn[i][x]].tolist() for x in range(t)])
        for j in range(t):  # 逆序处理
            xj_id = ptnn[i][j]
            xj_vector = data_embedding[xj_id]

            dist = np.linalg.norm(xj_vector - xi_vector)  # ---------直径

            if dist == radius_max:  # 如果直径相同, 直接计算圆心
                middle_vector = xj_vector  # 圆心
            else:
                middle_vector = (radius_max / dist) * (np.array(xj_vector - xi_vector)) + xi_vector  # 相对近邻图的圆心
            coo_1.append(middle_vector.tolist())
            inners = list()  # 当前近邻点所获得的内部rng点
            for x in range(t):
                dist_xk_xi = np.linalg.norm(xi_vector - matrix_tnn[x])  # 这两个距离都要小于 radius_max
                dist_xk_xj = np.linalg.norm(middle_vector - matrix_tnn[x])  # 这两个距离都要小于 radius_max
                if dist_xk_xi <= radius_max and dist_xk_xj <= radius_max:
                    inners.append(ptnn[i][x])
            if not (ptnn[i][j] in inners):
                inners.append(ptnn[i][j])
            """保存一个近邻数据点的 RNG 点列表-----------------------------------"""
            inners_l.append(inners)
        """保存每个点的tnn的ARNG点的子团"""
        diver_tnn_fine.append(inners_l)
        coo.append(coo_1)  # 添加边线圆心

        """将有真子集集的散点列表进行合并，即把有交集的diver_top_l进行合并，使方向上合并"""
        # print(f'\ndata point {i}, 的k近邻: {ptnn[i]}')
        # print(f'data point: {i} 的近邻 inners 分布: {inners_l}')
        inners_union_2 = merge_sub_arng(t, inners_l)
        # print(f'合并后的inners: {inners_union_2}')

        """角度： 根据合并的散列点列表，计算xi的各个大方向散列的角度--------------------------------------------------"""
        angle_mean = []
        for nodes in inners_union_2:
            if len(nodes) > 1:
                comb2 = itertools.permutations(nodes, 2)
                for s2 in comb2:
                    angel_1 = angel_calculation(i, s2, data_embedding)
                    angle_mean.append(angel_1)
        angle_mean = np.mean(np.array(angle_mean))

        """ 各个角度和与系数积为xi的总角度: xi 到它的tnn的中心距离，xi的tnn球的最大半径."""
        tnn_r = [ptnd[i][-1]]
        bias = [bias_mid[i]]
        for p in ptnn[i]:
            tnn_r.append(ptnd[p][-1])
            bias.append(bias_mid[p])

        beta2 = 1 - ptnd[i][-1] / np.sum(np.array(tnn_r))  # tnn范围内半径归一化
        beta3 = 1 - bias_mid[i] / np.sum(np.array(bias))  # tnn范围内中心偏移归一化

        angle.append(angle_mean)
        angle_refine.append(angle_mean * beta2 * beta3)
    # print(f'angle: {angle}')
    # print(f'angle refined {angle_refine}')
    return diver_tnn_fine, angle, angle_refine, coo


def box_plot(nums):
    """  被检测的变量 放置在 nums[0], 检验num[0]在nums数列中是否为离群值，这里只检测(海拔的)上离群点
    """
    a = np.percentile(nums, (25, 75))
    iqr = a[1] - a[0]
    upper_bound = a[1] + 1.5 * iqr
    # lower_bound = a[0] - 1.5 * iqr

    if nums[0] > upper_bound:
        return 1
    else:
        return 0


def box_plot_clustering(nums):
    """  被检测的变量 放置在 nums[0], 检验num[0]在nums数列中是否为离群值，这里只检测(海拔的)上离群点
    """
    a = np.percentile(nums, (25, 75))
    iqr = a[1] - a[0]
    upper_bound = a[1] + 1.5 * iqr
    lower_bound = a[0] - 1.5 * iqr

    if nums[0] > upper_bound:
        flag = 1
    elif nums[0] < lower_bound:
        flag = -1
    else:
        flag = 0

    return flag


def tnn_box_plot(nums):
    a = np.percentile(nums, (25, 75))
    iqr = a[1] - a[0]
    upper_bound = a[1] + 1.5 * iqr
    # lower_bound = a[0] - 1.5 * iqr

    ff = list()
    for x in nums[1:]:
        if x > upper_bound:
            ff.append(1)
        else:
            ff.append(0)
    return ff


def preradius_nn(data_size, data_embedding, mean_radius, ptnn, ptnd, pttnn, pttnn_data):
    """
    本函数用于处理每个数据点的指定阈值内的近邻。 即当xi的tnn球半径小于全局半径均值时，则使用均值半径判别 xi的边界性
    """
    zhid_dnn = list()  # 指定距离内的近邻
    zhid_dnd = list()  # 指定距离内的近邻的对应距离
    for i in range(data_size, ):
        if ptnd[i][-1] >= mean_radius:
            zhid_dnn.append(ptnn[i])
            zhid_dnd.append(ptnd[i])
        else:
            temp = np.linalg.norm(np.array(pttnn_data[i]) - data_embedding[i], axis=1, keepdims=True).reshape(1, -1)
            simi_list = temp[0]
            simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序

            kns0, kvs0 = list(), list()
            for j in simi_sorted:
                if simi_list[j] <= mean_radius:
                    kns0.append(pttnn[i][j])  # 数据点xi的局部区域的第j个邻居点
                    kvs0.append(simi_list[j])
                else:
                    break
            zhid_dnn.append(kns0[1:])  # 去除节点自身作为tnn成员
            zhid_dnd.append(kvs0[1:])  # 去除节点自身作为tnn成员

    return zhid_dnn, zhid_dnd


def absolute_boundary_point(i, data_embedding, ptnn, mid, zhid_dnn):
    """先计算xi与它的tnn/k中心点的反向延展点"""
    vi = data_embedding[i]
    vm = mid[i]
    vw = (vi - vm) * 2 + vm
    vec_wi = vi - vw  # division

    # 先计算 xi的外延点与xi之间的点会随着xi变为1类边界点而成为1类边界点
    length_wi = np.linalg.norm(vec_wi)
    inner_proximity = []
    for p in ptnn[i]:  # 夹角为锐角，并且距离小于length_wi的tnn成员xp
        x_p = data_embedding[p]
        vec_pw = x_p - vw
        length_pw = np.linalg.norm(vec_pw)
        if np.dot(vec_wi, vec_pw) > 0 and length_pw <= length_wi:
            inner_proximity.append(p)

    flag = True
    k = len(zhid_dnn[i])
    for j in range(k):
        x = zhid_dnn[i][j]
        reversenn = i in zhid_dnn[x]
        if reversenn:
            vj = data_embedding[x]
            vec_wj = vj - vw
            if np.dot(vec_wj, vec_wi) < 0:
                flag = False
                break
    return flag, vw, inner_proximity


def boundary_points(data_size, data_embedding, ptnd, ptnn, mid, pttnn, pttnn_data):
    """利用xi和它的不同tnn的中心点，计算数据点xi 是哪些数据点的边界点"""
    # boundary_nodes_1表示绝对边界点，boundary_nodes_2 表示潜在边界点
    boundaries = []
    boundary_flag = np.zeros(data_size)
    extension_p = []  # 保存用于计算 第1类边界点 的球内tnn/k中心点的反向点 即 mid[i]向 xi 的反向延展点 vw

    mean_radius = np.mean(np.array([ptnd[i][-1] for i in range(data_size)]))

    zhid_dnn, zhid_dnd = preradius_nn(data_size, data_embedding, mean_radius, ptnn, ptnd, pttnn, pttnn_data)  # 更新小于全局均值半径的球半径和半径内的近邻点列表

    for i in range(data_size):
        # """计算第1类边界点: 绝对边界点－－－－－－－－－"""
        flag, vw, inner_proximity = absolute_boundary_point(i, data_embedding, ptnn, mid, zhid_dnn)
        # print(f'flag: {flag}')
        extension_p.append(vw.tolist())
        if flag:
            boundaries.append(i)
            boundary_flag[i] = 1  # 表示边界点
            if len(inner_proximity) >= 1:
                for p in inner_proximity:
                    if boundary_flag[p] == 0:
                        boundaries.append(p)
                        boundary_flag[p] = 1
    return boundaries, extension_p, mean_radius, boundary_flag


def altitude(data_size, data_embedding, ptnn, ptnd, t, rnns):
    """ 这个函数用于计算每个数据点的海拔或高度，使用了基于相互近邻的点级系数和基于共享近邻的领域，融入了基于tnn的区域级系数"""
    ptnd = np.array(ptnd)  # 将二维列表进行数组化
    # """权重权重＊＊＊＊＊＊＊＊＊＊＊＊＊＊"""
    # weight = np.exp(ptnd) / np.sum(np.exp(ptnd), axis=1, keepdims=True)  # 大小为： 点数 × KNN
    weight = ptnd / np.sum(ptnd, axis=1, keepdims=True)  # 大小为： 点数 × KNN
    # w_indu_norm = np.array(in_degree) / (np.max(np.array(in_degree)) + 1)
    alpha = list()
    shared_nn = list()
    for i in range(data_size):  # 读取所有节点中的一个节点
        # """计算 xi与xj互为近邻，且共享近邻数量大于等于: 1/2 """
        height = []
        snn_temp = list()
        for j in range(t):
            xj = ptnn[i][j]
            xj_indice = xj
            shares = list(set(ptnn[i]).intersection(ptnn[xj_indice]))
            snn_temp.append(len(shares))

            if rnns[i][j] == 1:
                if len(shares) > 0:
                    sum0 = 0.0
                    for xk in shares:
                        sum0 = sum0 + np.linalg.norm(data_embedding[i] - data_embedding[xk]) + np.linalg.norm(data_embedding[xj_indice] - data_embedding[xk])  # dist(Ai-Xi)+dist(Aj-Xi)
                    height.append(sum0 / len(shares))  # 节点Xi与KNN中的某一个节点之间的共享近邻距离和/共享近邻点数，即共享近邻平均距离
                else:
                    height.append(2 * np.linalg.norm(data_embedding[i] - data_embedding[xj_indice]))  # 用 dist(Ai-Aj)  # 来构成一个更长的距离，说明该点距离Xi更远
            else:
                height.append(2 * np.linalg.norm(data_embedding[i] - data_embedding[xj_indice]))  # 用 dist(Ai-Aj)  # 来构成一个更长的距离，说明该点距离Xi更远

        alpha.append(np.dot(weight[i], np.array(height)))
        shared_nn.append(snn_temp)

    mean_alpha, std_alpha = [], []
    for i in range(data_size):
        tt = [alpha[i]]
        tt.extend([alpha[x] for x in ptnn[i]])
        mean_alpha.append(np.mean(np.array(tt)))
        std_alpha.append(np.std(np.array(tt), ddof=1))

    # """便于显示，这里进行了四舍五入运算－－－－－－－－－－－－－－－－－－－－－"""
    alpha = np.around(np.array(alpha), 4)
    return alpha, mean_alpha, std_alpha, shared_nn


def boxplot_alpha_and_min_angle(data_size, ptnn, alpha, t, cluster_label, rnns, angle_refine):
    """根据海拔计算各个数据点在自己的k近邻海拔内是否为离群海拔----1）如果在自己的tnn范围内，xi为海拔离群点，则xi必为离群点； 2）否则计算xi与它的tnn之间的海拔离群关系(统计关系)"""
    fbp, self_bp, dynamic_bp = [], [], []

    """step 1: 初始化 反向bp上离群，初始化值为－1，这里的－1不代表下离群－－－－－－－－－"""
    rbp = np.zeros((data_size, t)) - 1

    min_angle, min_angle_flag = [], []
    """step 2: 计算xi的上离群性和tnn成员的上离群性， 并更新近邻的反向上离群性－－－－－－－－"""
    for i in range(data_size):
        """计算局部区域内角度最小点和标记"""
        temp_l = [angle_refine[i]]
        temp_l.extend([angle_refine[j] for j in ptnn[i]])
        if angle_refine[i] == np.min(np.array(temp_l)):
            min_angle.append(i)
            min_angle_flag.append(1)
        else:
            min_angle_flag.append(0)

        ti_alphas = [alpha[i]]
        ti_alphas.extend([alpha[x].tolist() for x in ptnn[i]])
        self_bp.append(box_plot(ti_alphas))  # 保存当前点的自身离群性，即在自己的tnn范围内里否为上离群点
        ffbp = tnn_box_plot(ti_alphas)
        fbp.append(tnn_box_plot(ti_alphas))  # 保存邻居点的反向上离群，inver_bp
        for j in range(t):
            if rnns[i][j] == 1:
                temp1 = ptnn[ptnn[i][j]].tolist()
                xi_reverindex = temp1.index(cluster_label[i])
                rbp[ptnn[i][j]][xi_reverindex] = ffbp[j]
    return fbp, rbp, self_bp, min_angle, min_angle_flag


def pauta_criterion_alpha(data_size, alpha, ptnn, t, fbp, rbp, threshold_stability, shared_nn):
    mean_t = np.zeros((data_size)) - 1
    variance_t = np.zeros((data_size)) - 1
    fpc, rpc, pc_bi = [], [], []
    stability_pc = []
    coarse_points, stability_factors = [], []
    valuable_nn = []
    outlier = []

    for i in range(data_size):
        alpha_i = alpha[i]
        valuable_nn_temp = []
        """# 计算近邻海拔的 方差和均值"""
        if mean_t[i] == -1 and variance_t[i] == -1:
            xi_k_alphas = [alpha_i]
            xi_k_alphas.extend([alpha[x] for x in ptnn[i]])
            mean_t[i] = np.mean(np.array(xi_k_alphas))  # 均值
            variance_t[i] = np.std(np.array(xi_k_alphas), ddof=1)  # 方差

        """# 计算xi近邻海拔的 拉依达法则的正向值"""
        fpctemp, rpctemp, pctemp, cp_temp = [], [], [], []  # 临时变量： 拉依达法则的正向值 和反向值
        for j in range(t):
            xj = ptnn[i][j]
            alpha_j = alpha[xj]
            if alpha_j - mean_t[i] <= 2 * variance_t[i]:
                fpctemp.append(1)
            else:
                fpctemp.append(0)

            """# 计算xj近邻海拔的 拉依达法则的均值和方差"""
            if mean_t[xj] == -1 or variance_t[xj] == -1:
                xj_k_alphas = [alpha[xj]]
                xj_k_alphas.extend([alpha[x] for x in ptnn[xj]])
                mean_t[xj] = np.mean(np.array(xj_k_alphas))  # 均值
                variance_t[xj] = np.std(np.array(xj_k_alphas), ddof=1)  # 方差

            """# 计算xi近邻海拔的 拉依达法则的反向值"""
            if alpha_i - mean_t[xj] <= 2 * variance_t[xj]:
                rpctemp.append(1)
            else:
                rpctemp.append(0)

            """双向拉依达法则关系"""
            if fpctemp[j] == 1 and rpctemp[j] == 1:
                pctemp.append(1)
            else:
                pctemp.append(0)

            """根据四个特征，计算粗糙点集 coarse points"""
            # if pctemp[j] == 1 and fbp[i][j] == 0 and rbp[i][j] == 0 and shared_nn[i][j]/k >= 0.4:
            if pctemp[j] == 1 and fbp[i][j] == 0 and rbp[i][j] == 0:
                cp_temp.append(1)
                valuable_nn_temp.append(xj)
            else:
                cp_temp.append(0)

        valuable_nn.append(valuable_nn_temp)
        tt = np.mean(np.array(cp_temp))
        stability_factors.append(tt)
        if 0 < tt <= threshold_stability:
            coarse_points.append(i)
        elif tt == 0:
            outlier.append(i)
        fpc.append(fpctemp)
        rpc.append(rpctemp)
        pc_bi.append(pctemp)
        stability_pc.append(np.mean(np.array(pctemp)))
    return fpc, rpc, pc_bi, stability_pc, coarse_points, stability_factors, valuable_nn, outlier, mean_t, variance_t


def vote_mechanism(i, j, k, pknn, mean_t, variance_t, center, alpha):
    dbv = -1
    same, diff = 0, 0
    if i in pknn[j]:
        hid = pknn[j].tolist().index(i)
        for m in pknn[j][:hid+1]:
            if center[m] == center[i]:
                flag_sknn = len(set(pknn[i]).intersection(pknn[m])) >= int(k/2)
                if alpha[i] - mean_t[m] <= 3 * variance_t[m] and alpha[m] - mean_t[i] <= 3 * variance_t[i] and flag_sknn:
                    same += 1
                else:
                    diff += 1
        dbv = same - diff
    return dbv


def Detect_catchment_basin(data_size, cluster_label, alpha, stability_factors, threshold_stability, mean_t, variance_t, t, k, pknn, boundary_flag):
    node_and_altitude_ = list(zip(cluster_label, alpha))
    node_and_altitude = sorted(node_and_altitude_, key=lambda x: x[1], reverse=False)
    points = [item[0] for item in node_and_altitude]  # 临时保存排序后的节点列表

    center = [-1 for i in range(data_size)]  # 点的初始中心为-1
    basins, basins_p, competition, basin_competition = [], [], [], []
    basin_dict = dict()
    competition_flag = np.zeros(data_size)

    links, edge_links, edges = [], [], []
    basin_link, edge_basin_link = [], []
    basin_general_link = []
    valid_knn = [list() for i in range(data_size)]

    """所有糙点数据不作为主聚类点, -------只能被聚类点*****************"""

    """detecting catchment basin"""
    for i in points:
        """不处理糙点, 糙点参与聚类"""
        if stability_factors[i] <= threshold_stability:
            continue
        if center[i] == -1:
            center[i] = i  # 修改盆点
            basin_dict[i] = len(basin_dict)  # 新集水盆-->字典项
            basins.append(list())  # 新集水盆 -->列表
            basins[basin_dict[i]].append(i)  # 新集水盆 --> 添加新成员，盆点
            basins_p.append(list())  # 潜在空间 --> 新集水盆
            basins_p[basin_dict[i]].append(i)   # 潜在空间 --> 新集水盆字典项
            basin_competition.append(list())
            basin_general_link.append(list())

        """---------------开始聚类------------------"""
        Bid = basin_dict[center[i]]

        for m in range(t):
            j = pknn[i][m]
            """糙点： 只能被聚类，不能作为 沉点 ---------------------------------- """
            if stability_factors[j] > threshold_stability:
                if len(set(pknn[i]).intersection(pknn[j])) >= int(k/2) and alpha[i] - mean_t[j] <= 3 * variance_t[j] and alpha[j] - mean_t[i] <= 3 * variance_t[i]:
                    """记录点i的可用knn"""
                    valid_knn[i].append(j)  # 记录可用knn
                    if center[j] == -1:
                        center[j] = center[i]  # 修改 盆点
                        basins[Bid].append(j)
                        basins_p[Bid].append(j)
                        """添加可视化边 -------------------------------------------------------------------- """
                        edges.append((str(i), str(j)))

                    elif center[j] >= 0:
                        if center[i] != center[j] and boundary_flag[j] == 0:
                            if not(j in basins_p[Bid]):
                                basins_p[Bid].append(j)  # 不能竞争，但可以作为潜在空间的内容
                            if not(j in basin_competition[Bid]):
                                basin_competition[Bid].append(j)
                            if not(i in basin_competition[Bid]):
                                basin_competition[Bid].append(i)  # 样本i也是自身所有集水盆的竞争点
                            # basin_competition[basin_dict[center[j]]].append(j)  # 减少计算量，这里不进行对称处理
                            """competition 用于 可视化"""
                            if competition_flag[j] != 1:
                                competition.append(j)
                                competition_flag[j] = 1
                            if competition_flag[i] != 1:
                                competition.append(i)
                                competition_flag[i] = 1

                            f1 = i in basin_dict.keys()
                            f2 = j in basin_dict.keys()
                            if f1 or f2:
                                basin_link.append((i, j, min(alpha[i], alpha[j])))
                                edge_basin_link.append((str(i), str(j)))  # """ 可视化 集水盆之间的 连通------------------- """
                            else:
                                links.append((i, j, min(alpha[i], alpha[j]), center[i], center[j]))  # """ 为了减少计算时间， 只保留一个 集水盆 之间 的 连通 """
                                basin_general_link[Bid].append((i, j))
                                """可视化 --- 对称关系"""
                                edge_links.append((str(i), str(j)))  # """ 可视化 集水盆之间的 连通------------------- """

        for m in range(t, k):
            j = pknn[i][m]
            if stability_factors[j] > threshold_stability:
                if len(set(pknn[i]).intersection(pknn[j])) >= int(k/2) and alpha[i] - mean_t[j] <= 3 * variance_t[j] and alpha[j] - mean_t[i] <= 3 * variance_t[i]:
                    score = vote_mechanism(i, j, k, pknn, mean_t, variance_t, center, alpha)
                    if score > 0:
                        """记录点i的可用knn"""
                        valid_knn[i].append(j)  # 记录可用knn
                        """聚类xi 和 xj"""
                        if center[j] == -1:
                            center[j] = center[i]  # 修改 盆点
                            basins[Bid].append(j)
                            basins_p[Bid].append(j)
                            """添加可视化边 -------------------------------------------------------------------- """
                            edges.append((str(i), str(j)))

                        elif center[j] >= 0:
                            if center[i] != center[j]:
                                if not (j in basins_p[Bid]):
                                    basins_p[Bid].append(j)  # 不能竞争，但可以作为潜在空间的内容
                                if not (j in basin_competition[Bid]):
                                    basin_competition[Bid].append(j)
                                if not (i in basin_competition[Bid]):
                                    basin_competition[Bid].append(i)  # 样本i也是自身所有集水盆的竞争点
                                # basin_competition[basin_dict[center[j]]].append(j)  # 减少计算量，这里不进行对称处理
                                """competition 用于 可视化"""
                                if competition_flag[j] != 1:
                                    competition.append(j)
                                    competition_flag[j] = 1
                                if competition_flag[i] != 1:
                                    competition.append(i)
                                    competition_flag[i] = 1
                                f1 = i in basin_dict.keys()
                                f2 = j in basin_dict.keys()
                                if f1 or f2:
                                    basin_link.append((i, j, min(alpha[i], alpha[j])))
                                    edge_basin_link.append((str(i), str(j)))  # """ 可视化 集水盆之间的 连通------------------- """
                                else:
                                    links.append((i, j, min(alpha[i], alpha[j]), center[i], center[j]))  # """ 为了减少计算时间， 只保留一个 集水盆 之间 的 连通 """
                                    basin_general_link[Bid].append((i, j))
                                    """可视化 --- 对称关系"""
                                    edge_links.append((str(i), str(j)))  # """ 可视化 集水盆之间的 连通------------------- """

    unlabeled = [x for x in points if center[x] < 0]
    return center, points, basins, basin_dict, basins_p, links, edges, edge_links, competition, basin_competition, competition_flag, unlabeled, basin_link, edge_basin_link, valid_knn, basin_general_link


def mergeable_interval(basins_p, alpha):
    basin_median_alpha, basin_min_alpha = [], []
    for bps in basins_p:
        temp = [alpha[x] for x in bps]
        basin_median_alpha.append(np.median(np.array(temp)))
        basin_min_alpha.append(np.min(np.array(temp)))
    return basin_median_alpha, basin_min_alpha


def Merge_catchment_basin(center, basins, basins_p, basin_link, basin_dict, links, alpha, basin_competition, stability_factors, angle_refine):
    """
    step 1: 检测无效集水盆连通：首先将集水盆所有的点的ARNG角度进行盆内最大值归范化，如果xi为非边界点且归范化角度<=0.2，选择xi往盆点方向的kNN, 否则选择xi集水盆内ptNN, --->
    step 2: 利用 盆点连通聚类实现集水盆的聚类. 集水盆的连通关系： (i, j, min(alpha[i], alpha[j]), j, center[i], center[j])
    step 3: 在不可靠盆间连通的约束下，合并集水盆
    """
    """第1步： 优先利用包含盆点的盆间连通来合并集水盆×××××××××××××××××××××××××××××××××××××××××××"""
    edge_merge_basins = []
    edge_merge_points = []
    basin_median_alpha, basin_min_alpha = mergeable_interval(basins_p, alpha)
    for con in basin_link:
        i = con[0]
        j = con[1]
        mm = con[2]
        c1 = center[i]
        c2 = center[j]
        if c1 != c2:
            if mm <= basin_median_alpha[basin_dict[c1]] or mm <= basin_median_alpha[basin_dict[c2]]:
                if c1 <= c2:
                    host_basin = c1
                    host_point = i
                    guest_basin = c2
                else:
                    host_basin = c2
                    host_point = j
                    guest_basin = c1

                edge_merge_points.append((str(i), str(j)))
                edge_merge_basins.append((str(c1), str(c2)))

                """下面的集水盆内点的合并，用于更新集水盆内点的盆点标签"""
                for x in basins[basin_dict[guest_basin]]:
                    center[x] = center[host_point]
                basins[basin_dict[host_basin]].extend(basins[basin_dict[guest_basin]])

    """第2步： 计算每个集水盆内不能产生可靠盆间连通的节点××××××××××××××××××××××××××××××××××××××"""
    weaks = []
    basin_weak_point = []  # 记录集水盆内不可用于盆间连通的点----
    for i in range(len(basins)):
        if len(basin_competition[i]) > 0:
            # imax = max([angle_refine[x] for x in basins_p[i]])
            imax_2factors = max([angle_refine[x] * stability_factors[x] for x in basins_p[i]])
            """记录集水盆内 弱角度点------"""
            tt = [x for x in basin_competition[i] if angle_refine[x] / imax_2factors <= 0.2]
            basin_weak_point.append(tt)
            weaks.extend([str(t) for t in tt])   # -------------------------------------------------------------------------------
        else:
            basin_weak_point.append([])

    """第3步：在不可靠盆间连通的约束下，合并集水盆××××××××××××××××××××××××××××××××××××××××××××××××××××"""
    links_sorted = sorted(links, key=lambda x: x[2], reverse=False)
    for con in links_sorted:
        i = con[0]
        j = con[1]
        mm = con[2]
        c1 = center[i]
        c2 = center[j]
        if c1 != c2:
            if i in basin_weak_point[basin_dict[con[3]]] or i in basin_weak_point[basin_dict[con[4]]] or j in basin_weak_point[basin_dict[con[3]]] or j in basin_weak_point[basin_dict[con[4]]]:
                continue
            else:
                if mm <= basin_median_alpha[basin_dict[c1]] or mm <= basin_median_alpha[basin_dict[c2]]:
                    if c1 <= c2:
                        host_basin = c1
                        host_point = i
                        guest_basin = c2
                    else:
                        host_basin = c2
                        host_point = j
                        guest_basin = c1

                    edge_merge_points.append((str(i), str(j)))
                    edge_merge_basins.append((str(c1), str(c2)))

                    """下面的集水盆内点的合并，用于更新集水盆内点的盆点标签"""
                    for x in basins[basin_dict[guest_basin]]:
                        center[x] = center[host_point]
                    basins[basin_dict[host_basin]].extend(basins[basin_dict[guest_basin]])
    """第3步，结束×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××"""

    return center, edge_merge_points, edge_merge_basins, weaks


def assignment_remaining_points(points, center, pknn, mean_t, variance_t, alpha, k, boundary_flag):
    edge_assignment = []
    for j in points:
        if center[j] < 0:
            for i in pknn[j]:
                if len(set(pknn[i]).intersection(pknn[j])) >= int(k/2) and alpha[i] - mean_t[j] <= 3 * variance_t[j] and alpha[j] - mean_t[i] <= 3 * variance_t[i] and center[i] >= 0:
                    center[j] = center[i]
                    edge_assignment.append((str(i), str(j)))
                    break
                else:
                    if boundary_flag[j] == 1:
                        for i in pknn[j]:
                            if len(set(pknn[i]).intersection(pknn[j])) / k >= 0.5 and center[i] >= 0:
                                center[j] = center[i]
                                edge_assignment.append((str(i), str(j)))
                                break

    return center, edge_assignment


def wc_ca_clustering(data_embedding, t, k, threshold_stability):
    cluster_label, pos, node_dict, data_size = read_dataset(data_embedding)
    large_k = k
    large_knn, large_knd = large_nn(data_embedding, large_k, data_size)
    ptnn, ptnd, pknn, pknd, bias_mid, mid, midtnn, midtnd, rnns, rnn_c, diver, direction_tnn, pttnn, pttnn_data, loc_rnn = preprocess(data_size, data_embedding, t, k, large_knn, large_knd)

    """diver_tnn_fine 三层列表：(data_size * 外层散度点 * 每个散度的内部节点列表), diver_tnn_union 是三层列表 (data_size * 合并后的散度top * 每个散度的内部节点列表)"""
    """angle_diver 是一个 data_size * 1的形状，保存每个点的tnn内的散列角度－－－－－－－－－－"""
    diver_tnn_fine, angle, angle_refine, coo = anti_relative_neighborhodd_graph(data_size, data_embedding, ptnn, t, ptnd, bias_mid)

    """计算每个数据点的tnn内的散列角度－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－"""
    alpha, mean_alpha, std_alpha, shared_nn = altitude(data_size, data_embedding, ptnn, ptnd, t, rnns)

    """---------------------------------------------------------------------------------"""
    fbp, rbp, self_bp, min_angle, min_angle_flag = boxplot_alpha_and_min_angle(data_size, ptnn, alpha, t, cluster_label, rnns, angle_refine)

    fpc, rpc, pc_bi, stability_pc, coarse_points, stability_factors, valuable_nn, outlier, mean_t, variance_t = pauta_criterion_alpha(data_size, alpha, ptnn, t, fbp, rbp, threshold_stability, shared_nn)  # 基于海拔的拉依达法则

    boundaries, extension_p, mean_radius, boundary_flag = boundary_points(data_size, data_embedding, ptnd, ptnn, mid, pttnn, pttnn_data)

    center, points, basins, basin_dict, basins_p, links, edges, edge_links, competition, basin_competition, competition_flag, unlabeled, basin_link, edge_basin_link, valid_knn, basin_general_link = Detect_catchment_basin(data_size, cluster_label, alpha, stability_factors, threshold_stability, mean_t, variance_t, t, k, pknn, boundary_flag)

    center, edge_merge_points, edge_merge_basins, weaks = Merge_catchment_basin(center, basins, basins_p, basin_link, basin_dict, links, alpha, basin_competition, stability_factors, angle_refine)

    center, edge_assignment = assignment_remaining_points(points, center, pknn, mean_t, variance_t, alpha, k, boundary_flag)

    return center


if __name__ == '__main__':
    path = 'datasets/'
    # file_name = 'Aggregation.txt'
    # file_name = 'CMC.txt'
    # file_name = 'Compound.txt'
    # file_name = 'D31.txt'
    # file_name = 'flame.txt'
    # file_name = 'jain.txt'
    file_name = 'pathbased.txt'
    # file_name = 'R15.txt'
    # file_name = 'spiral.txt'
    # file_name = 's2.txt'

    # file_name = 'ecoli.txt'
    # file_name = 'movement_libras.txt'
    # file_name = 'ionosphere.txt'
    # file_name = 'iris.txt'
    # file_name = 'seeds.txt'
    # file_name = 'segmentation.txt'
    # file_name = 'wdbc.txt'
    # file_name = 'wine.txt'
    # file_name = 'spectrometer.txt'
    # file_name = 'glass.txt'
    # file_name = 'OlivettiFaces.txt'
    # file_name = 'usps.txt'
    # file_name = 'mnist.txt'


    data = np.loadtxt(path + file_name, delimiter=',')

    data_embedding = data[:, :-1]
    truth = data[:, -1]

    t = 6
    k = 16
    threshold_stability = 0.4

    center = wc_ca_clustering(t, k, threshold_stability)

    ari = metrics.adjusted_rand_score(center, truth)
    ami = metrics.adjusted_mutual_info_score(center, truth)
    fmi = metrics.fowlkes_mallows_score(center, truth)
    noise_ratio = len([x for x in center if x < 0]) / len(truth)
    print(f't: {t}, k: {k}, theta: {threshold_stability}, ARI: {ari}, AMI: {ami}, FMI: {fmi}, noise_ratio: {noise_ratio}')


