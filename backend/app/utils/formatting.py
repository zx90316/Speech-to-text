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


