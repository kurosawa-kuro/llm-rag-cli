def split_text(text, chunk_size=500, overlap=100):
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # 単語境界で分割: 後方の最寄り空白を探す
        boundary = text.rfind(" ", start, end)
        if boundary <= start:
            # chunk_size 内に空白がない場合、前方の空白まで拡張
            boundary = text.find(" ", end)
            if boundary == -1:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

        chunks.append(text[start:boundary].strip())
        # 次のチャンク開始位置: overlap 分戻るが、単語境界を保持
        next_start = boundary + 1 - overlap
        if next_start <= start:
            next_start = boundary + 1
        # next_start が単語の途中なら次の空白まで進む
        if next_start > 0 and next_start < len(text) and text[next_start - 1] != " ":
            word_start = text.rfind(" ", start, next_start)
            if word_start > start:
                next_start = word_start + 1
            else:
                next_start = boundary + 1
        start = next_start

    return chunks


def split_by_structure(text, chunk_size=None, overlap=100):
    import re

    if not text:
        return []

    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if chunk_size is None:
        return paragraphs

    result = []
    for para in paragraphs:
        if len(para) > chunk_size:
            result.extend(split_text(para, chunk_size=chunk_size, overlap=overlap))
        else:
            result.append(para)
    return result
