import torch
import torchaudio
import os
import numpy as np


def echo_cancellation(audio, sample_rate, filter_length=256):  # 缩短滤波器长度，使其影响范围更小
    """
    使用自适应滤波器去除回声，添加了防溢出处理，处理更保守

    参数:
    audio (torch.Tensor): 输入音频张量
    sample_rate (int): 采样率
    filter_length (int): 滤波器长度，已调整为更短的值

    返回:

    torch.Tensor: 去除回声后的音频
    """
    # 确保音频为单声道
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # 转换为numpy数组进行信号处理
    audio_np = audio.squeeze().numpy()

    # 防止信号过大导致溢出，先归一化
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * 0.9  # 缩放至安全范围

    # 初始化自适应滤波器参数
    mu = 0.0005 # 学习率减小10倍，使滤波器更新更慢，处理更保守
    h = np.zeros(filter_length, dtype=np.float64)  # 使用更高精度的浮点数
    y = np.zeros_like(audio_np, dtype=np.float64)  # 输出信号
    e = np.zeros_like(audio_np, dtype=np.float64)  # 误差信号（去回声后的信号）

    # 假设回声是延迟的信号，这里使用简单的延迟作为参考信号
    delay = int(sample_rate * 0.1)  # 100ms延迟
    x = np.zeros_like(audio_np, dtype=np.float64)
    x[delay:] = audio_np[:-delay] if delay < len(audio_np) else audio_np

    # 应用LMS自适应滤波器
    for n in range(len(audio_np)):
        # 获取当前输入窗口
        x_window = np.zeros(filter_length, dtype=np.float64)
        start = max(0, n - filter_length + 1)
        x_window[filter_length - (n - start + 1):] = x[start:n + 1]

        # 滤波器输出
        y[n] = np.dot(h, x_window)

        # 限制输出范围，防止溢出
        y[n] = np.clip(y[n], -1.0, 1.0)

        # 计算误差（期望信号 - 滤波器输出）
        e[n] = audio_np[n] - y[n]

        # 更新滤波器系数，更保守的更新
        h += mu * e[n] * x_window

        # 限制滤波器系数范围，增加限制使变化更小
        h = np.clip(h, -0.5, 0.5)  # 系数范围缩小，使滤波器影响减弱

    # 转换回torch张量前先归一化
    max_e = np.max(np.abs(e))
    if max_e > 0:
        e = e / max_e * 0.9

    return torch.from_numpy(e).unsqueeze(0).float()


def process_audio(input_file, output_file, volume_multiplier=5.0):
    """
    处理音频：放大指定倍数并去除回声

    参数:
    input_file (str): 输入音频文件路径
    output_file (str): 输出处理后音频文件路径
    volume_multiplier (float): 音量放大倍数，默认为5.0
    """
    try:
        # 加载音频文件
        wav, sr = torchaudio.load(input_file)

        # 音量放大指定倍数
        wav = wav * volume_multiplier

        # 去除回声
        wav = echo_cancellation(wav, sr)

        # 保存处理后的音频
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torchaudio.save(output_file, wav.cpu(), sample_rate=sr)

        print(f"音频已放大{volume_multiplier}倍并去除回声，已保存至: {output_file}")

    except Exception as e:
        print(f"处理音频时出错: {e}")
        import traceback
        traceback.print_exc()


def batch_process_audio(input_dir, output_dir, volume_multiplier=4.0):
    """
    批量处理目录中的所有音频文件

    参数:
    input_dir (str): 输入音频目录
    output_dir (str): 输出音频目录
    volume_multiplier (float): 音量放大倍数
    """
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 支持的音频文件扩展名
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    # 获取所有音频文件
    audio_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and os.path.splitext(f)[1].lower() in audio_extensions]

    if not audio_files:
        print(f"在目录 {input_dir} 中未找到音频文件")
        return

    print(f"找到 {len(audio_files)} 个音频文件，开始处理...")

    # 处理每个音频文件
    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        file_name, file_ext = os.path.splitext(audio_file)
        output_path = os.path.join(output_dir, f"{file_name}_processed{file_ext}")

        print(f"\n处理文件: {audio_file}")
        process_audio(input_path, output_path, volume_multiplier)


if __name__ == "__main__":
    # 配置路径
    input_dir = r"D:\声纹识别测试\2"# 替换为你的输入目录
    output_dir = r"D:\声纹识别测试\2"   # 替换为你的输出目录
    volume_multiplier = 5.0  # 音量放大倍数


    # 处理路径格式
    input_dir = input_dir.replace('\\', '/')
    output_dir = output_dir.replace('\\', '/')

    batch_process_audio(input_dir, output_dir, volume_multiplier)

