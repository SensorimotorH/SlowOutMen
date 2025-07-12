import open3d as o3d
import os
from typing import Tuple
from SlowOutMen.VisionUnderstanding.PoseEstimation.registration import *
from typing import Any, Dict, List, Union
current_dir = os.path.dirname(os.path.abspath(__file__))


class PoseEstimator:
    """
    位姿估计器

    用于估计部分点云与完整点云之间的位姿

    1. 加载部分点云和完整点云
    2. 预处理点云
    3. 执行全局配准
    4. 执行ICP精配准
    5. 执行GICP配准
    6. 获取最终变换矩阵
    """

    def __init__(self, partial_point_cloud: Union[o3d.geometry.PointCloud, str], object_name: str, debug: bool = False) -> None:
        """
        初始化位姿估计器
        :param partial_point_cloud: 部分点云数据o3d.geometry.PointCloud 或点云文件路径
        :param object_name: 物体名称,根据此名称加载完整点云
        :param debug: 是否启用调试模式
        """
        if isinstance(partial_point_cloud, str):
            self.partial_pcd = o3d.io.read_point_cloud(partial_point_cloud)
        elif isinstance(partial_point_cloud, o3d.geometry.PointCloud):
            self.partial_pcd = partial_point_cloud
        else:
            raise TypeError(
                "partial_point_cloud must be a string or o3d.geometry.PointCloud")

        # 处理完整点云输入
        obj_ply_path = os.path.join(
            current_dir, "obj_ply", f"{object_name}.ply")
        try:
            self.complete_pcd = o3d.io.read_point_cloud(obj_ply_path)
        except:
            raise FileNotFoundError(f"找不到{object_name}的完整点云文件")
        self.debug = debug

        # 验证点云是否为空
        if len(self.partial_pcd.points) == 0:
            raise ValueError("部分点云为空")
        if len(self.complete_pcd.points) == 0:
            raise ValueError("完整点云为空")
    def _numpy_to_pointcloud(self, points):
        """
        将numpy数组转换为Open3D点云对象
        :param points: numpy数组，形状为(N, 3)或(N, 6)，其中N是点的数量
                    如果是(N, 3)，表示只有xyz坐标
                    如果是(N, 6)，表示xyz坐标和rgb颜色
        :return: Open3D点云对象
        """
        pcd = o3d.geometry.PointCloud()

        # 检查输入维度
        if points.shape[1] not in [3, 6]:
            raise ValueError("输入点云数组必须是(N,3)或(N,6)的形状")

        # 设置点坐标
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # 如果有颜色信息，设置颜色
        if points.shape[1] == 6:
            colors = points[:, 3:] / 255.0 if points[:,
                                                     3:].max() > 1 else points[:, 3:]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def __call__(self):

        return get_transformation_matrix(self.partial_pcd, self.complete_pcd, self.debug)

    def preprocess_point_cloud(self, pcd_pre: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Any, Any]:
        """
        点云预处理:估计法线、计算FPFH特征
        :param pcd: 输入点云
        :return: 点云、法线、FPFH特征
        """
        pcd, _ = pcd_pre.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        # 估计法线
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 计算FPFH特征
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100))
        if fpfh is None or fpfh.data.size == 0:
            raise RuntimeError("FPFH特征计算失败")
        print("FPFH特征计算完成")

        return pcd, pcd.normals, fpfh

    def execute_global_registration(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, source_fpfh: Any, target_fpfh: Any) -> Any:
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
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        print("全局配准完成")
        print(f"初始变换矩阵:\n{result.transformation}")
        return result

    def refine_registration(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, initial_transformation: Any) -> Any:
        """
        执行ICP精配准
        :param source: 源点云
        :param target: 目标点云
        :param initial_transformation: 初始变换矩阵
        :return: 精配准结果
        """
        distance_threshold = 0.06
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print("ICP精配准完成")
        print(f"ICP变换矩阵:\n{result.transformation}")
        return result

    def execute_gicp_registration(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, initial_transformation: Any) -> Any:
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
        print(f"GICP变换矩阵:\n{result.transformation}")
        return result

    def get_transformation_matrix(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, Debug: bool = False) -> Any:
        """
        获取最终变换矩阵
        :param source: 源点云
        :param target: 目标点云
        :param Debug: 是否启用调试模式
        :return: 最终变换矩阵
        """
        # 加载基准点云和待配准的分割点云
        target_pcd = target
        source_pcd = source

        # 点云预处理
        source, _, source_fpfh = self.preprocess_point_cloud(
            source_pcd)
        target, _, target_fpfh = self.preprocess_point_cloud(
            target_pcd)

        # 全局配准获取初始变换矩阵
        initial_result = self.execute_global_registration(
            source, target, source_fpfh, target_fpfh)
        initial_transformation = initial_result.transformation

        # 使用ICP进行精配准
        refined_result = self.refine_registration(
            source, target, initial_transformation)
        icp_transformation = refined_result.transformation

        # 使用GICP进行进一步优化
        gicp_result = self.execute_gicp_registration(
            source, target, icp_transformation)
        final_transformation = gicp_result.transformation

        print("最终变换矩阵（位姿）:")
        print(final_transformation)

        if Debug:
            source.transform(final_transformation)
            source.paint_uniform_color([1, 0, 0])
            target.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([source, target])

        return final_transformation

    def pose_estimate(self) -> Any:
        return self.get_transformation_matrix(self.partial_pcd, self.complete_pcd, self.debug)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    partial_cloud_path = os.path.join(current_dir, "test_partition_ply", "apple_partition.ply")
    complete_cloud_path = os.path.join(current_dir, "obj_ply", "apple.ply")

    partial_point_cloud = o3d.io.read_point_cloud(partial_cloud_path)
    object_name = "apple"

    # 创建位姿估计器
    pose_estimator = PoseEstimator(partial_point_cloud, object_name)
    result = pose_estimator()

    # 打印结果
    # print(f"完整点云转换到目标位姿的转换矩阵为：\n{result}")
    # print("\n配准评估:")
    # print(f"RANSAC配准得分: {pose_estimator.ransac_result.fitness}")
    # print(f"RANSAC内点RMSE: {pose_estimator.ransac_result.inlier_rmse}")
    # print(f"ICP配准得分: {pose_estimator.icp_result.fitness}")
    # print(f"ICP内点RMSE: {pose_estimator.icp_result.inlier_rmse}")
