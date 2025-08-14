from __future__ import annotations


def to_srt_time_format(seconds: float) -> str:
    millisec = int((seconds - int(seconds)) * 1000)
    minutes, seconds_int = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{millisec:03}"


def generate_srt(transcription_data: list[dict]) -> str:
    srt_content: list[str] = []
    for i, segment in enumerate(transcription_data, 1):
        start_time = to_srt_time_format(float(segment.get("start", 0.0)))
        end_time = to_srt_time_format(float(segment.get("end", 0.0)))
        text = str(segment.get("text", ""))

        srt_content.append(str(i))
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")

    return "\n".join(srt_content)


def parse_hhmmss(time_str: str) -> float:
    """將 HH:MM:SS 或 MM:SS 或 SS 解析為秒數(float)。
    接受小數秒，例如 01:02:03.5。
    """
    if not time_str:
        return 0.0
    parts = str(time_str).split(":")
    parts = [p.strip() for p in parts if p.strip() != ""]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60 + s
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    raise ValueError("時間格式錯誤，需為 SS、MM:SS 或 HH:MM:SS")


