class TaskParser:
    def __init__(self):
        # 定义可能的属性集合
        self.prop_names = {"sheet", "disk", "carton"}
        self.cabinet_floors = {"second", "third", "fourth"}
        self.cabinet_sides = {"left", "right"}
        self.table_sides = {"left", "right"}
        self.directions = {"on ", "left ",
                           "right ", "front ", "behind ", "back ", "in "}
        self.objects = {
            "drawer", "apple", "cup",
            "clock", "kettle", "xbox", "bowl", "scissors",
            "plate", "book", "wood"
        }
        self.drawer_layer = {
            "top", "bottom"
        }

    def parse_task(self, instruction):
        """
        解析任务指令，返回包含关键信息的字典

        返回字典格式:
        Round 1: {
            'round': 1,
            'prop': str,                    # 要操作的物品名称
            'floor': str,                   # 层数
            'cabninet_side':str,            # 哪边的柜子
            'target_table': str             # 物品的目标位置（桌子位置描述）
        }

        Round 2: {
            'round': 2,
            'prop': str,                    # 要查找的物品名称
            'texture': str,                 # 物品的纹理描述
            'target_direction': str,        # 相对目标物体的方向
            'target_object': str            # 目标参考物体
        }

        Round 3: {
            'round': 3,
            'action': 'find_same_and_put',
            'layer_index': str,             # 抽屉层数
            'target_direction': str,        # 目标位置的方向
            'target_object': str            # 目标位置的参考物体
        }
        """

        # 首先判断是哪一轮的任务
        if "Take the" in instruction and "from" in instruction:
            return self._parse_round1(instruction)
        elif "Find the" in instruction and "with" in instruction:
            return self._parse_round2(instruction)
        elif "Find another prop as same as" in instruction:
            return self._parse_round3(instruction)
        else:
            raise ValueError("Unknown instruction format")

    def _parse_round1(self, instruction):
        """解析第一轮任务：从柜子取物品放到桌子上"""
        result = {
            'round': 1,
        }

        # 提取物品名称
        words = instruction.split()
        prop_idx = words.index("the") + 1
        result['prop'] = words[prop_idx]

        # 提取源位置（柜子位置）
        source_start = instruction.index("from") + 5
        source_end = instruction.index("and put")
        source_text = instruction[source_start:source_end].strip()

        # 解析柜子位置的具体信息
        for floor in self.cabinet_floors:
            if floor in source_text:
                result['floor'] = floor
                break

        for side in self.cabinet_sides:
            if side in source_text:
                result['cabinet_side'] = side
                break

        # 提取目标位置（桌子位置）
        target_start = instruction.index("put it") + 7
        target_text = instruction[target_start:].strip()
        for side in self.table_sides:
            if side in target_text:
                result['target_table'] = side
                break

        return result

    def _parse_round2(self, instruction):
        """解析第二轮任务：找到特定纹理的物品并放置"""
        result = {
            'round': 2,
        }

        # 提取物品名称
        words = instruction.split()
        prop_idx = words.index("the") + 1
        result['prop'] = words[prop_idx]

        # 提取纹理描述
        texture_start = instruction.index("with") + 5
        texture_end = instruction.index("and put")
        result['texture'] = instruction[texture_start:texture_end].strip()

        # 提取目标方向和物体
        target_part = instruction[instruction.index("put it") + 7:].strip()
        for direction in self.directions:
            if direction in target_part:
                result['target_direction'] = direction
                object_part = target_part[target_part.index(
                    direction) + len(direction):].strip()
                for object in self.objects:
                    if object in object_part:
                        result['target_object'] = object
                break

        return result

    def _parse_round3(self, instruction):
        """解析第三轮任务：找到相同物品并放置"""
        result = {
            'round': 3,
        }

        # 提取柜子层数
        reference_part = instruction[instruction.index(
            "in the") + 7:instruction.index("layer")].strip()
        for index in self.drawer_layer:
            if index in reference_part:
                result['layer_index'] = index
                break

        # 提取目标方向和物体
        target_part = instruction[instruction.index("put it") + 7:].strip()
        for direction in self.directions:
            if direction in target_part:
                result['target_direction'] = direction
                object_part = target_part[target_part.index(
                    direction) + len(direction):].strip()
                for object in self.objects:
                    if object in object_part:
                        result['target_object'] = object
                break

        return result


if __name__ == "__main__":
    from Communication.ros2_mmk2_api import MMK2_Receiver
    import rclpy
    rclpy.init()
    receiver = MMK2_Receiver()
    task_parser = TaskParser()
    instruction = receiver.get_taskinfo()
    result = task_parser.parse_task(instruction)
    print(result)
