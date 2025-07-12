import open3d as o3d
import numpy as np


def load_point_cloud(file_path):
    """
    加载点云文件
    :param file_path: 点云文件路径
    :return: open3d点云对象
    """
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"加载点云文件：{file_path}")
    print(f"点云包含 {len(pcd.points)} 个点")
    return pcd


def preprocess_point_cloud(pcd_pre):
    """
    点云预处理：估计法线、计算FPFH特征
    :param pcd: 输入点云
    :return: 点云、法线、FPFH特征
    """
    pcd, _ = pcd_pre.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    # 估计法线
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # print("法线估计完成")

    # 计算FPFH特征
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100))
    if fpfh is None or fpfh.data.size == 0:
        raise RuntimeError("FPFH特征计算失败")
    print("FPFH特征计算完成")

    return pcd, pcd.normals, fpfh


def execute_global_registration(source, target, source_fpfh, target_fpfh):
    """
    执行全局配准
    :param source: 源点云
    :param target: 目标点云
    :param source_fpfh: 源点云FPFH特征
    :param target_fpfh: 目标点云FPFH特征
    :return: 初始配准结果
    """
    distance_threshold = 0.15
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    print("全局配准完成")
    print(f"初始变换矩阵：\n{result.transformation}")
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, initial_transformation):
    """
    执行ICP精配准
    :param source: 源点云
    :param target: 目标点云
    :param source_fpfh: 源点云FPFH特征
    :param target_fpfh: 目标点云FPFH特征
    :param initial_transformation: 初始变换矩阵
    :return: 精配准结果
    """
    distance_threshold = 0.06
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print("ICP精配准完成")
    print(f"ICP变换矩阵：\n{result.transformation}")
    return result


def execute_gicp_registration(source, target, initial_transformation):
    """
    执行GICP配准
    :param source: 源点云
    :param target: 目标点云
    :param initial_transformation: 初始变换矩阵
    :return: GICP配准结果
    """
    distance_threshold = 0.075
    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
    print("GICP配准完成")
    print(f"GICP变换矩阵：\n{result.transformation}")
    return result


def get_transformation_matrix(source, target, Debug=False):
    """
    获取最终变换矩阵
    :param source: 源点云或路径
    :param target: 目标点云或路径
    :return: 最终变换矩阵
    """
    # 加载基准点云（由MeshLab生成的工件点云）和待配准的分割点云
    target_pcd = target
    # target_pcd = load_point_cloud("../output/point_clouds/point_cloud_8.ply")  # 基准点云
    source_pcd = source  # 分割点云
    # 缩放因子
    # scale_factor = 1

    # # 使用 scale 方法缩放点云
    # target_pcd.scale(scale_factor, center=target_pcd.get_center())
    # 点云预处理
    source, source_normals, source_fpfh = preprocess_point_cloud(source_pcd)
    target, target_normals, target_fpfh = preprocess_point_cloud(target_pcd)

    # 全局配准获取初始变换矩阵
    initial_result = execute_global_registration(
        source, target, source_fpfh, target_fpfh)
    initial_transformation = initial_result.transformation

    # 使用ICP进行精配准
    refined_result = refine_registration(
        source, target, source_fpfh, target_fpfh, initial_transformation)
    icp_transformation = refined_result.transformation

    # 使用GICP进行进一步优化
    gicp_result = execute_gicp_registration(source, target, icp_transformation)
    final_transformation = gicp_result.transformation

    # 输出最终变换矩阵（即模型的位姿）
    print("最终变换矩阵（位姿）：")
    print(final_transformation)

    if Debug:
        # 可视化配准结果
        source.transform(final_transformation)
        source.paint_uniform_color([1, 0, 0])  # 红色为变换后的源点云
        target.paint_uniform_color([0, 1, 0])  # 绿色为目标点云
        o3d.visualization.draw_geometries([source, target])

    return final_transformation


def main():

    # 加载基准点云（由MeshLab生成的工件点云）和待配准的分割点云
    target_pcd = load_point_cloud(
        "SlowOutMen/VisionUnderstanding/PoseEstimation/obj_ply/timeclock.ply")
    # target_pcd = load_point_cloud("../output/point_clouds/point_cloud_8.ply")  # 基准点云
    source_pcd = load_point_cloud(
        "SlowOutMen/VisionUnderstanding/PoseEstimation/test_partition_ply/clock_partition.ply")  # 分割点云
    # # 缩放因子
    # scale_factor = 1

    # # 使用 scale 方法缩放点云
    # target_pcd.scale(scale_factor, center=target_pcd.get_center())
    # 点云预处理
    source, source_normals, source_fpfh = preprocess_point_cloud(source_pcd)
    target, target_normals, target_fpfh = preprocess_point_cloud(target_pcd)

    # 全局配准获取初始变换矩阵
    initial_result = execute_global_registration(
        source, target, source_fpfh, target_fpfh)
    initial_transformation = initial_result.transformation

    # 使用ICP进行精配准
    refined_result = refine_registration(
        source, target, source_fpfh, target_fpfh, initial_transformation)
    icp_transformation = refined_result.transformation

    # 使用GICP进行进一步优化
    gicp_result = execute_gicp_registration(source, target, icp_transformation)
    final_transformation = gicp_result.transformation

    # 输出最终变换矩阵（即模型的位姿）
    print("最终变换矩阵（位姿）：")
    print(final_transformation)

    # 可视化配准结果
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    source.transform(final_transformation)
    source.paint_uniform_color([1, 0, 0])  # 红色为变换后的源点云
    target.paint_uniform_color([0, 1, 0])  # 绿色为目标点云
    o3d.visualization.draw_geometries([source, target, coord_frame])


if __name__ == "__main__":
    main()
