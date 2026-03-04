from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mcap.writer import Writer
from google.protobuf.descriptor_pb2 import FileDescriptorSet

# --- Foxglove native schemas (protobuf) ---
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
# NOTE: foxglove-schemas-protobuf==0.2.2 没有 Timestamp_pb2。
# Foxglove 的 protobuf schemas 实际使用的是 google.protobuf.Timestamp。
from google.protobuf.timestamp_pb2 import Timestamp

from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Color_pb2 import Color

from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField


# -----------------------------------------------------------------------------
# 输入格式约定（你接真实数据时严格按这里来）
# -----------------------------------------------------------------------------
# 对齐键：key_frame_id
# - str。用来表示“同一帧”。你可以用递增帧号("0","1",...)或 uuid。
#
# 时间：timestamp_ns
# - int，纳秒。用于写入到 MCAP 的 log_time / publish_time。
# - 同一 key_frame_id 下，points 与 boxes 最好同一个 timestamp_ns；
#   如果存在抖动，可以用 timestamp_tolerance_ns 做最近邻匹配。
#
# 坐标系：frame_id
# - str，例如："lidar_top" / "lidar_left" / "base_link" / "map"。
# - add_points/add_box 时都要显式传入。
#
# 点云：points
# - np.ndarray, shape=(N,3) or (N,4)
#   - (N,3): [x,y,z]
#   - (N,4): [x,y,z,intensity]
# - dtype 推荐 float32（内部会转换）
#
# 3D box：box（逐个录入）
# - np.ndarray, shape=(7,) / (1,7) / (7,1)
# - 含义固定为：
#   [cx, cy, cz, length, width, height, yaw]
# - yaw: 绕 +Z 旋转（右手系），单位 rad
# - track_id/class_id: 可选（int 或 str）
#
# 颜色规则：
# - 同时有 track_id 和 class_id：按 class_id 上色
# - 只有一个：按存在的那个上色
# - 都没有：固定色
#
# 外参（TF）：add_extrinsic_4x4(parent, child, T_parent_child)
# - parent/child: str，坐标系名字
# - T_parent_child: np.ndarray shape=(4,4)
#   p_parent = T_parent_child @ [p_child, 1]
# -----------------------------------------------------------------------------


def now_ns() -> int:
    return int(time.time() * 1e9)


def ts_from_ns(t_ns: int) -> Timestamp:
    """Convert ns -> google.protobuf.Timestamp.
    google.protobuf.Timestamp 字段名是 seconds / nanos。
    """
    s = t_ns // 1_000_000_000
    ns = t_ns % 1_000_000_000
    return Timestamp(seconds=int(s), nanos=int(ns))


def quat_from_yaw(yaw: float) -> Quaternion:
    half = 0.5 * float(yaw)
    return Quaternion(x=0.0, y=0.0, z=float(np.sin(half)), w=float(np.cos(half)))


def make_descriptor_set(msg_cls) -> bytes:
    """Build FileDescriptorSet bytes for MCAP schema registration."""
    fds = FileDescriptorSet()
    file_desc = msg_cls.DESCRIPTOR.file

    seen = set()
    stack = [file_desc]
    files = []
    while stack:
        fd = stack.pop()
        if fd.name in seen:
            continue
        seen.add(fd.name)
        files.append(fd)
        for dep in fd.dependencies:
            stack.append(dep)

    for fd in files:
        fd.CopyToProto(fds.file.add())
    return fds.SerializeToString()


def rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3, 3)

    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


@dataclass
class _BoxRec:
    box7: np.ndarray  # (7,)
    frame_id: str
    track_id: Optional[str] = None
    class_id: Optional[str] = None


@dataclass
class _FrameBuf:
    """Buffered data for a key_frame_id."""

    timestamp_ns: int
    key_frame_id: str

    # 多路点云：frame_id -> points
    pointclouds: Dict[str, np.ndarray] = field(default_factory=dict)

    # 多个 box
    boxes: List[_BoxRec] = field(default_factory=list)


class McapTool:
    def __init__(
        self,
        out_path: str,
        *,
        timestamp_tolerance_ns: int = 0,
        tf_topic: str = "/tf",
        fixed_box_color: Tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.9),
        required_pointcloud_frames: Optional[List[str]] = None,
    ):
        """Create an MCAP writer.

        required_pointcloud_frames:
            - None: 只要有任意 1 路点云 + 至少 1 个 box，就认为该帧可写。
            - List[str]: 必须这些 frame 的点云都到齐 + 至少 1 个 box，才认为该帧可写。

        你说的“close() 之前保证都对齐”：
        - close() 会先尝试 flush 所有能 flush 的帧
        - 如果仍有帧缺点云/缺 box/缺必需点云帧，会 raise Exception
        """

        self.out_path = str(Path(out_path).expanduser())
        self.timestamp_tolerance_ns = int(timestamp_tolerance_ns)
        self.tf_topic = tf_topic
        self.fixed_box_color = fixed_box_color
        self.required_pointcloud_frames = required_pointcloud_frames

        self._frames_by_ts: Dict[int, _FrameBuf] = {}
        self._frames_by_key: Dict[str, _FrameBuf] = {}

        self._f = open(self.out_path, "wb")
        self._w = Writer(self._f)
        self._w.start()

        # schemas
        self._sid_pointcloud = self._w.register_schema(
            name="foxglove.PointCloud",
            encoding="protobuf",
            data=make_descriptor_set(PointCloud),
        )
        self._sid_sceneupdate = self._w.register_schema(
            name="foxglove.SceneUpdate",
            encoding="protobuf",
            data=make_descriptor_set(SceneUpdate),
        )
        self._sid_frametransform = self._w.register_schema(
            name="foxglove.FrameTransform",
            encoding="protobuf",
            data=make_descriptor_set(FrameTransform),
        )

        # channels
        self._ch_tf = self._w.register_channel(self.tf_topic, message_encoding="protobuf", schema_id=self._sid_frametransform)
        self._ch_boxes = self._w.register_channel("/perception/tracks_3d", message_encoding="protobuf", schema_id=self._sid_sceneupdate)

        # 注意：多路点云需要“每个 frame_id 一个 channel”，否则 topic 相同但 schema 相同也能写
        # 这里采用：topic 固定为 /lidar/points/<frame_id>
        self._ch_points_by_frame: Dict[str, int] = {}

    # ---------------- TF ----------------

    def add_extrinsic_4x4(
        self,
        *,
        timestamp_ns: int,
        parent_frame_id: str,
        child_frame_id: str,
        T_parent_child: np.ndarray,
    ) -> None:
        T = np.asarray(T_parent_child, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_parent_child must be (4,4), got {T.shape}")

        t = T[0:3, 3]
        qx, qy, qz, qw = rotmat_to_quat_xyzw(T[0:3, 0:3])

        msg = FrameTransform(
            timestamp=ts_from_ns(int(timestamp_ns)),
            parent_frame_id=str(parent_frame_id),
            child_frame_id=str(child_frame_id),
            translation=Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2])),
            rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )
        t_ns = int(timestamp_ns)
        self._w.add_message(self._ch_tf, t_ns, msg.SerializeToString(), publish_time=t_ns)

    # ---------------- data ----------------

    def add_points(
        self,
        *,
        timestamp_ns: int,
        key_frame_id: str,
        points: np.ndarray,
        frame_id: str,
    ) -> None:
        fb = self._get_or_create_frame(timestamp_ns=int(timestamp_ns), key_frame_id=str(key_frame_id))
        fb.pointclouds[str(frame_id)] = np.asarray(points)
        self._try_flush(fb)

    def add_box(
        self,
        *,
        timestamp_ns: int,
        key_frame_id: str,
        box: np.ndarray,
        frame_id: str,
        track_id: Optional[Union[int, str]] = None,
        class_id: Optional[Union[int, str]] = None,
    ) -> None:
        fb = self._get_or_create_frame(timestamp_ns=int(timestamp_ns), key_frame_id=str(key_frame_id))

        b = np.asarray(box).reshape(-1)
        if b.shape[0] != 7:
            raise ValueError(f"box must be 7D, got {np.asarray(box).shape}")
        if b.dtype != np.float32:
            b = b.astype(np.float32)

        fb.boxes.append(
            _BoxRec(
                box7=b,
                frame_id=str(frame_id),
                track_id=None if track_id is None else str(track_id),
                class_id=None if class_id is None else str(class_id),
            )
        )
        self._try_flush(fb)

    def close(self) -> None:
        """Flush everything possible, then ensure all frames are aligned."""

        # 1) 尝试 flush 所有可 flush 的帧
        for fb in list(self._frames_by_key.values()):
            self._try_flush(fb)

        # 2) 如果还有残留，说明存在未对齐帧 -> 报错
        if self._frames_by_key:
            problems = []
            for k, fb in self._frames_by_key.items():
                missing = []
                if len(fb.boxes) == 0:
                    missing.append("boxes")
                if not fb.pointclouds:
                    missing.append("pointcloud")
                if self.required_pointcloud_frames:
                    for req in self.required_pointcloud_frames:
                        if req not in fb.pointclouds:
                            missing.append(f"pointcloud:{req}")
                problems.append(f"{k} (ts={fb.timestamp_ns}) missing: {', '.join(missing) if missing else 'unknown'}")

            raise RuntimeError(
                "Unaligned frames remain before close().\n" + "\n".join(problems)
            )

        self._w.finish()
        self._f.close()

    # ---------------- alignment ----------------

    def _get_or_create_frame(self, *, timestamp_ns: int, key_frame_id: str) -> _FrameBuf:
        if timestamp_ns in self._frames_by_ts:
            fb = self._frames_by_ts[timestamp_ns]
            self._frames_by_key[fb.key_frame_id] = fb
            return fb

        if self.timestamp_tolerance_ns > 0 and self._frames_by_ts:
            nearest_ts = min(self._frames_by_ts.keys(), key=lambda t: abs(t - timestamp_ns))
            if abs(nearest_ts - timestamp_ns) <= self.timestamp_tolerance_ns:
                return self._frames_by_ts[nearest_ts]

        if key_frame_id in self._frames_by_key:
            fb = self._frames_by_key[key_frame_id]
            self._frames_by_ts[fb.timestamp_ns] = fb
            return fb

        fb = _FrameBuf(timestamp_ns=timestamp_ns, key_frame_id=key_frame_id)
        self._frames_by_ts[fb.timestamp_ns] = fb
        self._frames_by_key[fb.key_frame_id] = fb
        return fb

    def _is_frame_ready(self, fb: _FrameBuf) -> bool:
        if len(fb.boxes) == 0:
            return False
        if not fb.pointclouds:
            return False
        if self.required_pointcloud_frames:
            return all(req in fb.pointclouds for req in self.required_pointcloud_frames)
        return True

    def _try_flush(self, fb: _FrameBuf) -> None:
        if not self._is_frame_ready(fb):
            return

        t_ns = fb.timestamp_ns

        # 写 boxes（一个 SceneUpdate）
        su_msg = self._build_sceneupdate_boxes(t_ns, fb.boxes)
        self._w.add_message(self._ch_boxes, t_ns, su_msg.SerializeToString(), publish_time=t_ns)

        # 写多路点云（每个 frame_id 一个 topic）
        for pc_frame, points in fb.pointclouds.items():
            ch = self._get_or_create_pointcloud_channel(pc_frame)
            pc_msg = self._build_pointcloud(t_ns, points, pc_frame)
            self._w.add_message(ch, t_ns, pc_msg.SerializeToString(), publish_time=t_ns)

        # drop
        self._frames_by_ts.pop(fb.timestamp_ns, None)
        self._frames_by_key.pop(fb.key_frame_id, None)

    def _get_or_create_pointcloud_channel(self, frame_id: str) -> int:
        if frame_id in self._ch_points_by_frame:
            return self._ch_points_by_frame[frame_id]

        # 每个 frame 一个 topic，方便在 Foxglove 中单独开关
        topic = f"/lidar/points/{frame_id}"
        ch = self._w.register_channel(topic, message_encoding="protobuf", schema_id=self._sid_pointcloud)
        self._ch_points_by_frame[frame_id] = ch
        return ch

    # ---------------- builders ----------------

    def _build_pointcloud(self, t_ns: int, points: np.ndarray, frame_id: str) -> PointCloud:
        pts = np.asarray(points)
        if pts.dtype != np.float32:
            pts = pts.astype(np.float32)

        if pts.ndim != 2 or pts.shape[1] not in (3, 4):
            raise ValueError(f"points must be (N,3) or (N,4), got {pts.shape}")

        has_i = pts.shape[1] == 4
        data = pts.tobytes(order="C")
        stride = 16 if has_i else 12

        fields = [
            PackedElementField(name="x", offset=0, type=PackedElementField.FLOAT32),
            PackedElementField(name="y", offset=4, type=PackedElementField.FLOAT32),
            PackedElementField(name="z", offset=8, type=PackedElementField.FLOAT32),
        ]
        if has_i:
            fields.append(PackedElementField(name="intensity", offset=12, type=PackedElementField.FLOAT32))

        return PointCloud(
            timestamp=ts_from_ns(t_ns),
            frame_id=str(frame_id),
            point_stride=stride,
            fields=fields,
            data=data,
        )

    def _build_sceneupdate_boxes(self, t_ns: int, boxes: List[_BoxRec]) -> SceneUpdate:
        su = SceneUpdate()

        for i, rec in enumerate(boxes):
            cx, cy, cz, L, W, H, yaw = rec.box7.tolist()

            eid = rec.track_id if rec.track_id is not None else f"box_{i}"
            color = self._color_for_box(rec)

            ent = SceneEntity(
                timestamp=ts_from_ns(t_ns),
                frame_id=str(rec.frame_id),
                id=str(eid),
            )

            cube = CubePrimitive(
                pose=Pose(
                    position=Vector3(x=float(cx), y=float(cy), z=float(cz)),
                    orientation=quat_from_yaw(float(yaw)),
                ),
                size=Vector3(x=float(L), y=float(W), z=float(H)),
                color=color,
            )

            ent.cubes.append(cube)
            su.entities.append(ent)

        return su

    def _color_for_box(self, rec: _BoxRec) -> Color:
        if rec.class_id is not None and rec.track_id is not None:
            return self._color_from_key(rec.class_id)
        if rec.class_id is not None:
            return self._color_from_key(rec.class_id)
        if rec.track_id is not None:
            return self._color_from_key(rec.track_id)

        r, g, b, a = self.fixed_box_color
        return Color(r=float(r), g=float(g), b=float(b), a=float(a))

    @staticmethod
    def _color_from_key(key: str) -> Color:
        palette = [
            (1.0, 0.2, 0.2, 0.9),
            (0.2, 1.0, 0.2, 0.9),
            (0.2, 0.6, 1.0, 0.9),
            (1.0, 0.8, 0.2, 0.9),
            (0.8, 0.2, 1.0, 0.9),
            (0.2, 1.0, 0.9, 0.9),
            (0.9, 0.9, 0.2, 0.9),
        ]
        h = hash(key)
        r, g, b, a = palette[h % len(palette)]
        return Color(r=r, g=g, b=b, a=a)


# -----------------------------------------------------------------------------
# DEMO
# -----------------------------------------------------------------------------

def demo_generate(out_path: str = "demo_aligned_close.mcap"):
    # 要求 lidar_top 和 lidar_left 都到齐才 flush
    tool = McapTool(out_path, required_pointcloud_frames=["lidar_top", "lidar_left"])

    # 静态外参
    t0 = now_ns()
    T_base_lidar_top = np.eye(4)
    T_base_lidar_top[2, 3] = 1.7
    tool.add_extrinsic_4x4(timestamp_ns=t0, parent_frame_id="base_link", child_frame_id="lidar_top", T_parent_child=T_base_lidar_top)

    T_base_lidar_left = np.eye(4)
    T_base_lidar_left[0, 3] = 0.8
    T_base_lidar_left[1, 3] = 0.5
    T_base_lidar_left[2, 3] = 1.6
    tool.add_extrinsic_4x4(timestamp_ns=t0, parent_frame_id="base_link", child_frame_id="lidar_left", T_parent_child=T_base_lidar_left)

    for k in range(30):
        t = now_ns()
        key = str(k)

        # 两路点云
        pts_top = (np.random.randn(2000, 3).astype(np.float32) * np.array([10.0, 4.0, 0.5], dtype=np.float32))
        pts_left = (np.random.randn(2000, 3).astype(np.float32) * np.array([10.0, 4.0, 0.5], dtype=np.float32))

        tool.add_points(timestamp_ns=t, key_frame_id=key, points=pts_top, frame_id="lidar_top")
        tool.add_points(timestamp_ns=t, key_frame_id=key, points=pts_left, frame_id="lidar_left")

        # boxes
        for i in range(8):
            box7 = np.array(
                [
                    np.random.uniform(-20, 20),
                    np.random.uniform(-8, 8),
                    np.random.uniform(0.5, 1.5),
                    np.random.uniform(3.5, 5.0),
                    np.random.uniform(1.6, 2.2),
                    np.random.uniform(1.3, 2.0),
                    np.random.uniform(-3.14, 3.14),
                ],
                dtype=np.float32,
            )
            tool.add_box(timestamp_ns=t, key_frame_id=key, box=box7, frame_id="base_link", track_id=i, class_id=i % 5)

        time.sleep(0.01)

    # 关键：close() 会强制检查是否还有未对齐帧
    tool.close()
    print("Wrote", out_path)


if __name__ == "__main__":
    demo_generate()