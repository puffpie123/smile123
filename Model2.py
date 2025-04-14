import torch
import cv2
import numpy as np
import argparse
import copy
import time
import os
import threading
import queue
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics.nn.tasks import attempt_load_weights
from plate_recognition.plate_rec import get_plate_result, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from fonts.cv_puttext import cv2ImgAddText

# ------------------- 处理逻辑部分 -------------------

def allFilePath(rootPath, allFIleList):
    """读取文件夹内的文件，放到list"""
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)

def four_point_transform(image, pts):
    """透视变换得到车牌小图"""
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def letter_box(img, size=(640, 640)):
    """yolo 前处理 letter_box操作"""
    h, w, _ = img.shape
    r = min(size[0]/h, size[1]/w)
    new_h, new_w = int(h*r), int(w*r)
    new_img = cv2.resize(img, (new_w, new_h))
    left = int((size[1]-new_w)/2)
    top = int((size[0]-new_h)/2)
    right = size[1]-left-new_w
    bottom = size[0]-top-new_h
    img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top

def load_model(weights, device):
    """加载yolov8 模型"""
    model = attempt_load_weights(weights, device=device)  # load FP32 model
    return model

def xywh2xyxy(det):
    """xywh转化为xyxy"""
    y = det.clone()
    y[:, 0] = det[:, 0] - det[:, 2]/2
    y[:, 1] = det[:, 1] - det[:, 3]/2
    y[:, 2] = det[:, 0] + det[:, 2]/2
    y[:, 3] = det[:, 1] + det[:, 3]/2
    return y

def my_nums(dets, iou_thresh, device):
    """nms操作"""
    y = dets.clone()
    y_box_score = y[:, :5]
    index = torch.argsort(y_box_score[:, -1], descending=True)
    keep = []
    while index.numel() > 0:
        i = index[0].item()
        keep.append(i)
        if index.numel() == 1:
            break
        x1 = torch.maximum(y_box_score[i, 0], y_box_score[index[1:], 0])
        y1 = torch.maximum(y_box_score[i, 1], y_box_score[index[1:], 1])
        x2 = torch.minimum(y_box_score[i, 2], y_box_score[index[1:], 2])
        y2 = torch.minimum(y_box_score[i, 3], y_box_score[index[1:], 3])
        zero_ = torch.tensor(0.0).to(device)
        w = torch.maximum(zero_, x2 - x1)
        h = torch.maximum(zero_, y2 - y1)
        inter_area = w * h
        nuion_area1 = (y_box_score[i, 2] - y_box_score[i, 0]) * (y_box_score[i, 3] - y_box_score[i, 1])
        union_area2 = (y_box_score[index[1:], 2] - y_box_score[index[1:], 0]) * (y_box_score[index[1:], 3] - y_box_score[index[1:], 1])
        iou = inter_area / (nuion_area1 + union_area2 - inter_area)
        idx = torch.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(dets, r, left, top):
    """坐标还原到原图上"""
    dets[:, [0, 2]] = dets[:, [0, 2]] - left
    dets[:, [1, 3]] = dets[:, [1, 3]] - top
    dets[:, :4] /= r
    return dets

def post_processing(prediction, conf, iou_thresh, r, left, top, device):
    """后处理"""
    prediction = prediction.permute(0, 2, 1).squeeze(0)
    xc = prediction[:, 4:6].amax(1) > conf  # 过滤掉小于conf的框
    x = prediction[xc]
    if not len(x):
        return []
    boxes = x[:, :4]  # 框
    boxes = xywh2xyxy(boxes)  # 中心点 宽高 变为 左上 右下两个点
    score, index = torch.max(x[:, 4:6], dim=-1, keepdim=True)  # 找出得分和所属类别
    x = torch.cat((boxes, score, x[:, 6:14], index), dim=1)  # 重新组合
    score = x[:, 4]
    keep = my_nums(x, iou_thresh, device)
    x = x[keep]
    x = restore_box(x, r, left, top)
    return x

def pre_processing(img, opt, device):
    """前处理"""
    img, r, left, top = letter_box(img, (opt.img_size, opt.img_size))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img / 255.0
    img = img.unsqueeze(0)
    return img, r, left, top

def det_rec_plate(img, img_ori, detect_model, plate_rec_model, opt, device):
    """检测和识别车牌"""
    result_list = []
    img_tensor, r, left, top = pre_processing(img, opt, device)  # 前处理
    with torch.no_grad():
        predict = detect_model(img_tensor)[0]
    outputs = post_processing(predict, opt.conf, opt.iou_thresh, r, left, top, device)  # 后处理
    for output in outputs:
        result_dict = {}
        output = output.squeeze().cpu().numpy().tolist()
        rect = output[:4]
        rect = [int(x) for x in rect]
        label = output[-1]
        roi_img = img_ori[rect[1]:rect[3], rect[0]:rect[2]]
        if int(label):  # 判断是否是双层车牌，是双牌的话进行分割后然后拼接
            roi_img = get_split_merge(roi_img)
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model, is_color=True)

        result_dict['plate_no'] = plate_number  # 车牌号
        result_dict['plate_color'] = plate_color  # 车牌颜色
        result_dict['rect'] = rect  # 车牌roi区域
        result_dict['detect_conf'] = output[4]  # 检测区域得分
        result_dict['roi_height'] = roi_img.shape[0]  # 车牌高度
        result_dict['color_conf'] = color_conf  # 颜色得分
        result_dict['plate_type'] = int(label)  # 单双层 0单层 1双层
        result_list.append(result_dict)
    return result_list

def draw_result(orgimg, dict_list, is_color=False):
    """车牌结果画出来"""
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']

        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = int(0.05 * w)
        padding_h = int(0.11 * h)
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        result_p = result['plate_no']
        if result['plate_type'] == 0:  # 单层
            result_p += " " + result['plate_color']
        else:  # 双层
            result_p += " " + result['plate_color'] + "双层"
        result_str += result_p + " "

        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框

        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # 获得字体的大小
        label_width, label_height = labelSize[0]
        if rect_area[0] + label_width > orgimg.shape[1]:  # 防止显示的文字越界
            rect_area[0] = int(orgimg.shape[1] - label_width)
        orgimg = cv2.rectangle(
            orgimg,
            (rect_area[0], int(rect_area[1] - round(1.6 * label_height))),
            (int(rect_area[0] + round(1.2 * label_width)), rect_area[1] + label_height),
            (255, 255, 255),
            cv2.FILLED
        )  # 画文字框,背景白色

        if len(result) >= 6:
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * label_height)), (0, 0, 0), 21)

    print(result_str)
    return orgimg

def process_images(file_list, detect_model, plate_rec_model, opt, device, save_path, clors, callback=None):
    """处理图片文件夹中的所有图片"""
    count = 0
    time_all = 0
    time_begin = time.time()
    for pic_ in file_list:
        print(count, pic_, end=" ")
        time_b = time.time()  # 开始时间
        img = cv2.imread(pic_)
        if img is None:
            print("无法读取图片:", pic_)
            continue
        img_ori = copy.deepcopy(img)
        result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model, opt, device)
        time_e = time.time()
        ori_img = draw_result(img, result_list)  # 将结果画在图上
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path, img_name)  # 图片保存的路径
        time_gap = time_e - time_b  # 计算单个图片识别耗时
        if count:
            time_all += time_gap
        count += 1
        cv2.imwrite(save_img_path, ori_img)  # 保存结果图片
        if callback:
            callback(img_name, save_img_path)
    total_time = time.time() - time_begin
    avg_time = time_all / (len(file_list) - 1) if len(file_list) > 1 else time_all
    print(f"总耗时: {total_time:.2f} 秒, 平均每张图片耗时: {avg_time:.2f} 秒")
    if callback:
        callback("完成", f"总耗时: {total_time:.2f} 秒, 平均每张图片耗时: {avg_time:.2f} 秒")

def process_video(video_path, detect_model, plate_rec_model, opt, device, save_path, clors, frame_queue, callback=None, pause_event=None):
    """处理视频文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        if callback:
            callback("错误", f"无法打开视频文件: {video_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_video_path = os.path.join(save_path, f"{video_name}_result.mp4")
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

    frame_count = 0
    time_all = 0
    time_begin = time.time()

    while True:
        # 检查是否需要暂停
        if pause_event and not pause_event.is_set():
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break
        print(f"处理视频帧: {frame_count}", end="\r")
        time_b = time.time()
        img_ori = copy.deepcopy(frame)
        result_list = det_rec_plate(frame, img_ori, detect_model, plate_rec_model, opt, device)
        ori_img = draw_result(frame, result_list)  # 将结果画在图上
        out.write(ori_img)  # 写入视频
        time_e = time.time()
        time_gap = time_e - time_b
        if frame_count:
            time_all += time_gap
        frame_count += 1

        # 将处理后的帧放入队列
        if frame_queue is not None:
            frame_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            try:
                img_pil = img_pil.resize((600, 400), Image.Resampling.LANCZOS)
            except AttributeError:
                img_pil = img_pil.resize((600, 400), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img_pil)
            frame_queue.put(img_tk)

        if callback:
            callback("进度", f"处理帧: {frame_count}")

    cap.release()
    out.release()
    total_time = time.time() - time_begin
    avg_time = time_all / (frame_count - 1) if frame_count > 1 else time_all
    print(f"\n视频处理完成. 总帧数: {frame_count}, 总耗时: {total_time:.2f} 秒, 平均每帧耗时: {avg_time:.2f} 秒")
    if callback:
        callback("完成", f"视频处理完成. 总帧数: {frame_count}, 保存路径: {save_video_path}")

def process_single_image(image_path, detect_model, plate_rec_model, opt, device, save_path, clors, display_callback=None):
    """处理单张图片"""
    print("处理单张图片:", image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        if display_callback:
            display_callback(None, "无法读取图片。")
        return
    img_ori = copy.deepcopy(img)
    result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model, opt, device)
    ori_img = draw_result(img, result_list)  # 将结果画在图上
    img_name = os.path.basename(image_path)
    save_img_path = os.path.join(save_path, img_name)  # 图片保存的路径
    cv2.imwrite(save_img_path, ori_img)  # 保存结果图片
    print(f"处理完成，结果保存在: {save_img_path}")
    if display_callback:
        display_callback(save_img_path, "处理完成。")

# ------------------- GUI部分 -------------------

class PlateRecognitionGUI:
    def __init__(self, master):
        self.master = master
        master.title("车牌识别工具")
        master.geometry("800x800")  # 增加高度以适应视频帧显示
        master.resizable(False, False)

        # 初始化模型相关
        self.detect_model = None
        self.plate_rec_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # 输入文件路径
        self.input_path = StringVar()
        # 输出文件夹路径，默认设置为当前工作目录下的 'results' 文件夹
        default_output = os.path.join(os.getcwd(), "results")
        self.output_path = StringVar(value=default_output)
        # 输入类型
        self.input_type = StringVar(value="image")  # "image" 或 "video"
        # 参数
        self.conf_thresh = DoubleVar(value=0.3)
        self.iou_thresh = DoubleVar(value=0.5)
        self.img_size = IntVar(value=640)

        # 状态变量
        self.is_processing = False
        self.is_paused = False
        self.pause_event = threading.Event()
        self.pause_event.set()  # 初始为不暂停

        # 创建界面
        self.create_widgets()

        # 加载模型
        self.load_models()

        # 初始化队列用于视频帧传输
        self.frame_queue = queue.Queue()

        # 开始检查队列
        self.master.after(100, self.update_frame)

    def create_widgets(self):
        # 输入类型选择
        input_type_frame = LabelFrame(self.master, text="选择输入类型")
        input_type_frame.pack(padx=10, pady=5, fill="x")

        Radiobutton(input_type_frame, text="单张图片", variable=self.input_type, value="image").pack(side="left", padx=10, pady=5)
        Radiobutton(input_type_frame, text="单个视频", variable=self.input_type, value="video").pack(side="left", padx=10, pady=5)

        # 输入文件选择
        input_frame = Frame(self.master)
        input_frame.pack(padx=10, pady=5, fill="x")

        Label(input_frame, text="选择文件:").pack(side="left", padx=5)
        Entry(input_frame, textvariable=self.input_path, width=50).pack(side="left", padx=5)
        Button(input_frame, text="浏览", command=self.browse_input).pack(side="left", padx=5)

        # 输出文件夹选择
        output_frame = Frame(self.master)
        output_frame.pack(padx=10, pady=5, fill="x")

        Label(output_frame, text="输出文件夹:").pack(side="left", padx=5)
        Entry(output_frame, textvariable=self.output_path, width=50).pack(side="left", padx=5)
        Button(output_frame, text="浏览", command=self.browse_output).pack(side="left", padx=5)

        # 参数设置
        # param_frame = LabelFrame(self.master, text="参数设置")
        # param_frame.pack(padx=10, pady=5, fill="x")

        # Label(param_frame, text="置信度阈值:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        # Entry(param_frame, textvariable=self.conf_thresh, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Label(param_frame, text="IoU 阈值:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        # Entry(param_frame, textvariable=self.iou_thresh, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Label(param_frame, text="图像大小:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        # Entry(param_frame, textvariable=self.img_size, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # 开始/暂停按钮
        self.start_button = Button(self.master, text="开始识别", command=self.start_recognition, bg="green", fg="white")
        self.start_button.pack(pady=10)

        # 结果显示区域
        result_frame = LabelFrame(self.master, text="识别结果")
        result_frame.pack(padx=10, pady=5, fill="both", expand=True)

        # 对于视频，我们需要一个Label来显示帧
        self.result_label = Label(result_frame)
        self.result_label.pack(padx=5, pady=5)

        # 状态栏
        self.status = StringVar()
        self.status.set("等待输入")
        status_bar = Label(self.master, textvariable=self.status, bd=1, relief=SUNKEN, anchor=W)
        status_bar.pack(side="bottom", fill="x")

    def browse_input(self):
        if self.input_type.get() == "image":
            file_path = filedialog.askopenfilename(title="选择图片文件", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        else:
            file_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv")])
        if file_path:
            self.input_path.set(file_path)

    def browse_output(self):
        folder_selected = filedialog.askdirectory(title="选择输出文件夹")
        if folder_selected:
            self.output_path.set(folder_selected)

    def load_models(self):
        """加载检测和识别模型"""
        try:
            detect_model_path = 'weights/yolov8s.pt'  # 默认检测模型路径
            rec_model_path = 'weights/plate_rec_color.pth'  # 默认识别模型路径

            self.status.set("加载检测模型...")
            self.detect_model = load_model(detect_model_path, self.device)
            self.detect_model.eval()
            self.status.set("加载字符识别模型...")
            self.plate_rec_model = init_model(self.device, rec_model_path, is_color=True)
            self.status.set("模型加载完成。")
            print(f"检测模型和识别模型已加载。")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}")
            self.status.set("模型加载失败。")

    def start_recognition(self):
        """开始或暂停识别"""
        if not self.is_processing:
            # 开始识别
            input_path = self.input_path.get()
            output_path = self.output_path.get()
            input_type = self.input_type.get()
            conf = self.conf_thresh.get()
            iou = self.iou_thresh.get()
            img_size = self.img_size.get()

            if not input_path:
                messagebox.showwarning("警告", "请先选择输入文件。")
                return
            if not output_path:
                messagebox.showwarning("警告", "请先选择输出文件夹。")
                return

            # 配置参数
            class Opt:
                def __init__(self, img_size, conf, iou_thresh):
                    self.img_size = img_size
                    self.conf = conf
                    self.iou_thresh = iou_thresh

            opt = Opt(img_size=img_size, conf=conf, iou_thresh=iou)

            # 启动识别线程
            self.is_processing = True
            self.is_paused = False
            self.pause_event.set()  # 确保不暂停
            self.status.set("开始识别...")

            # 修改按钮为“暂停识别”
            self.start_button.config(text="暂停识别", command=self.pause_recognition)

            threading.Thread(target=self.run_recognition, args=(input_path, output_path, input_type, opt), daemon=True).start()

        else:
            if not self.is_paused:
                # 暂停识别
                self.pause_event.clear()
                self.is_paused = True
                self.status.set("识别已暂停。")
                self.start_button.config(text="继续识别", command=self.resume_recognition)
            else:
                # 继续识别
                self.pause_event.set()
                self.is_paused = False
                self.status.set("继续识别...")
                self.start_button.config(text="暂停识别", command=self.pause_recognition)

    def pause_recognition(self):
        """暂停识别"""
        # 功能已在 start_recognition 中实现
        pass

    def resume_recognition(self):
        """继续识别"""
        # 功能已在 start_recognition 中实现
        pass

    def run_recognition(self, input_path, save_path, input_type, opt):
        """运行识别逻辑"""
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if input_type == "image":
                # 处理单张图片
                process_single_image(input_path, self.detect_model, self.plate_rec_model, opt, self.device, save_path, self.clors, self.display_result)
                self.status.set("识别完成。")
                # 重置按钮
                self.start_button.config(text="开始识别", command=self.start_recognition)
                self.is_processing = False
            elif input_type == "video":
                # 处理视频
                process_video(input_path, self.detect_model, self.plate_rec_model, opt, self.device, save_path, self.clors, self.frame_queue, self.update_status, self.pause_event)
                self.status.set("识别完成。")
                # 重置按钮
                self.start_button.config(text="开始识别", command=self.start_recognition)
                self.is_processing = False
        except Exception as e:
            messagebox.showerror("错误", f"识别过程中出错: {e}")
            self.status.set("识别出错。")
            # 重置按钮
            self.start_button.config(text="开始识别", command=self.start_recognition)
            self.is_processing = False
            self.is_paused = False

    def display_result(self, save_img_path, message):
        """显示识别后的图片"""
        if save_img_path and os.path.exists(save_img_path):
            img = cv2.imread(save_img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # 修改这里：将 Image.ANTIALIAS 替换为 Image.Resampling.LANCZOS
            try:
                img_pil = img_pil.resize((600, 400), Image.Resampling.LANCZOS)
            except AttributeError:
                img_pil = img_pil.resize((600, 400), Image.ANTIALIAS)

            img_tk = ImageTk.PhotoImage(img_pil)
            self.result_label.configure(image=img_tk)
            self.result_label.image = img_tk
            self.status.set(message)
        else:
            self.status.set(message)

    def update_frame(self):
        """从队列中获取帧并显示"""
        try:
            while not self.frame_queue.empty():
                img_tk = self.frame_queue.get_nowait()
                self.result_label.configure(image=img_tk)
                self.result_label.image = img_tk
        except queue.Empty:
            pass
        # 每100ms检查一次队列
        self.master.after(100, self.update_frame)

    def update_status(self, status_type, message):
        """更新状态信息"""
        if status_type == "进度":
            self.status.set(message)
        elif status_type == "完成":
            self.status.set(message)
            messagebox.showinfo("完成", message)
        elif status_type == "错误":
            messagebox.showerror("错误", message)
            self.status.set("识别出错。")

# ------------------- 主程序 -------------------

def main():
    root = Tk()
    gui = PlateRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
