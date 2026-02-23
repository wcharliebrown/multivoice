#!/usr/bin/env python3
"""
epub_to_markdown.py

Reads an ePub file and:
  - Writes one Markdown file per chapter to chapters/
  - Writes chapters/characters.md with character names extracted from the text

Output filenames follow the Multivoice convention:
    chapters/00-Front-Matter.md
    chapters/01-Born-Yesterday.md
    chapters/02-Professional-Skeeball.md
    ...

After running this script, use Claude Code to convert the chapters to
screenplay format (see README for the prompt).

Usage:
    python epub_to_markdown.py book.epub
    python epub_to_markdown.py book.epub --output-dir ./chapters

Dependencies:
    pip install beautifulsoup4
"""

import argparse
import re
import sys
import warnings
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

try:
    from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    sys.exit("Missing dependency: pip install beautifulsoup4 lxml")


# ─── ePub parsing ─────────────────────────────────────────────────────────────

def _opf_path(zf: zipfile.ZipFile) -> str:
    container_xml = zf.read("META-INF/container.xml")
    root = ET.fromstring(container_xml)
    ns = {"cn": "urn:oasis:names:tc:opendocument:xmlns:container"}
    return root.find(".//cn:rootfile", ns).get("full-path")


def _read_zip_item(zf: zipfile.ZipFile, path: str) -> bytes:
    try:
        return zf.read(path)
    except KeyError:
        lower = path.lower()
        match = next((n for n in zf.namelist() if n.lower() == lower), None)
        if match:
            return zf.read(match)
        raise FileNotFoundError(f"Not found in epub: {path}")


def _ncx_titles(zf: zipfile.ZipFile, opf_dir: str, manifest: dict, opf_root) -> dict:
    opf_ns = {"opf": "http://www.idpf.org/2007/opf"}
    spine_el = opf_root.find(".//opf:spine", opf_ns)
    if spine_el is None:
        return {}
    toc_id = spine_el.get("toc")
    if not toc_id or toc_id not in manifest:
        return {}

    ncx_path = _join(opf_dir, manifest[toc_id]["href"])
    try:
        ncx_data = _read_zip_item(zf, ncx_path)
    except FileNotFoundError:
        return {}

    ncx_root = ET.fromstring(ncx_data)
    ncx_ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
    titles = {}
    for nav_point in ncx_root.findall(".//ncx:navPoint", ncx_ns):
        label = nav_point.find(".//ncx:text", ncx_ns)
        content = nav_point.find("ncx:content", ncx_ns)
        if label is not None and content is not None and label.text:
            src = unquote(content.get("src", "")).split("#")[0]
            titles[src] = label.text.strip()
    return titles


def _nav_titles(zf: zipfile.ZipFile, opf_dir: str, manifest: dict) -> dict:
    titles = {}
    for item in manifest.values():
        if "nav" not in item.get("properties", ""):
            continue
        nav_path = _join(opf_dir, item["href"])
        try:
            nav_data = _read_zip_item(zf, nav_path)
        except FileNotFoundError:
            continue
        soup = BeautifulSoup(nav_data, "lxml-xml")
        for a in soup.select("nav a"):
            href = unquote(a.get("href", "")).split("#")[0]
            text = a.get_text(strip=True)
            if href and text:
                titles[href] = text
    return titles


def _join(base: str, href: str) -> str:
    if not base:
        return href
    return str(Path(base) / href).lstrip("/")


def parse_epub(epub_path: str):
    """
    Returns (metadata, chapters).

    metadata: dict with keys title, creator, language, publisher, date
    chapters: list of {"title": str, "href": str, "html": str}
    """
    with zipfile.ZipFile(epub_path, "r") as zf:
        rootfile = _opf_path(zf)
        opf_dir = str(Path(rootfile).parent)
        if opf_dir == ".":
            opf_dir = ""

        opf_root = ET.fromstring(zf.read(rootfile))
        opf_ns = {
            "opf": "http://www.idpf.org/2007/opf",
            "dc":  "http://purl.org/dc/elements/1.1/",
        }

        metadata = {}
        for tag in ("title", "creator", "language", "publisher", "date"):
            el = opf_root.find(f".//dc:{tag}", opf_ns)
            metadata[tag] = el.text.strip() if el is not None and el.text else ""

        manifest = {}
        for item in opf_root.findall(".//opf:item", opf_ns):
            item_id = item.get("id")
            manifest[item_id] = {
                "href":       unquote(item.get("href", "")),
                "media-type": item.get("media-type", ""),
                "properties": item.get("properties", ""),
            }

        titles_by_href = _ncx_titles(zf, opf_dir, manifest, opf_root)
        titles_by_href.update(_nav_titles(zf, opf_dir, manifest))

        chapters = []
        for itemref in opf_root.findall(".//opf:itemref", opf_ns):
            idref = itemref.get("idref")
            if idref not in manifest:
                continue
            item = manifest[idref]
            mtype = item["media-type"]
            if "html" not in mtype and "xml" not in mtype:
                continue

            href = item["href"]
            item_path = _join(opf_dir, href)
            try:
                html = _read_zip_item(zf, item_path).decode("utf-8", errors="replace")
            except FileNotFoundError:
                continue

            title = titles_by_href.get(href, "")
            if not title:
                soup = BeautifulSoup(html, "lxml")
                for h in soup.find_all(["h1", "h2", "h3"]):
                    t = h.get_text(strip=True)
                    if t:
                        title = t
                        break

            chapters.append({"title": title, "href": href, "html": html})

    return metadata, chapters


# ─── HTML → Markdown ──────────────────────────────────────────────────────────

def _el_to_md(el) -> str:
    if isinstance(el, NavigableString):
        return str(el)
    if not isinstance(el, Tag):
        return ""

    tag = el.name or ""
    children = "".join(_el_to_md(c) for c in el.children)

    if tag == "h1":        return f"\n\n# {children.strip()}\n\n"
    if tag == "h2":        return f"\n\n## {children.strip()}\n\n"
    if tag == "h3":        return f"\n\n### {children.strip()}\n\n"
    if tag == "h4":        return f"\n\n#### {children.strip()}\n\n"
    if tag in ("h5","h6"): return f"\n\n##### {children.strip()}\n\n"
    if tag == "p":         return f"\n\n{children.strip()}\n\n"
    if tag == "br":        return "  \n"
    if tag == "hr":        return "\n\n---\n\n"
    if tag == "li":        return f"\n- {children.strip()}"
    if tag in ("ul","ol"): return f"\n{children}\n"
    if tag == "blockquote":
        lines = children.strip().split("\n")
        return "\n" + "\n".join(f"> {l}" for l in lines) + "\n"
    if tag in ("em","i"):     return f"*{children}*"
    if tag in ("strong","b"): return f"**{children}**"
    if tag == "code":         return f"`{children}`"
    if tag == "a":            return children
    if tag in ("script","style","head","meta","link"):
        return ""
    return children


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml-xml")
    body = soup.find("body") or soup
    md = _el_to_md(body)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ─── Front-matter detection ───────────────────────────────────────────────────

_FRONT_MATTER_RE = re.compile(
    r"\b(title[-\s]?page|copyright|contents|table[-\s]of[-\s]contents|toc|"
    r"dedication|foreword|preface|acknowledgements?|acknowledgments?|"
    r"about|cover|colophon|epigraph|half[-\s]?title|front[-\s]?matter|"
    r"navigation|nav)\b",
    re.IGNORECASE,
)


def _is_front_matter(chapter: dict) -> bool:
    title = chapter.get("title", "")
    href_stem = re.sub(r"[-_]", " ", Path(chapter.get("href", "")).stem)
    return bool(_FRONT_MATTER_RE.search(title) or _FRONT_MATTER_RE.search(href_stem))


# ─── Filename slug ────────────────────────────────────────────────────────────

def _slugify(title: str) -> str:
    """Convert a title to a filename-safe slug: 'Born Yesterday' → 'Born-Yesterday'.

    Strips a leading chapter number so ePub titles like '1 The Cube' or
    'Chapter 3: Marcus' don't produce double-numbered filenames like 01-1-The-Cube.
    """
    # Strip leading "1 ", "1- ", "Chapter 1: ", "Chapter 1 - " etc.
    title = re.sub(r'^(?:chapter\s+)?\d+[\s:\-\.]+', '', title, flags=re.IGNORECASE).strip()
    slug = re.sub(r"[^\w\s-]", "", title)
    slug = re.sub(r"[\s_]+", "-", slug.strip())
    return slug or "Untitled"


# ─── Writing chapters ─────────────────────────────────────────────────────────

def write_chapters(metadata: dict, chapters: list, output_dir: Path) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    front_seq = 0
    chap_seq = 0
    seen_chapter = False

    for chapter in chapters:
        md = html_to_markdown(chapter["html"])
        if not md.strip():
            continue

        title = chapter["title"]

        if not seen_chapter and _is_front_matter(chapter):
            front_seq += 1
            suffix = f"-{front_seq:02d}" if front_seq > 1 else ""
            slug = f"00-Front-Matter{suffix}"
            display_title = title or "Front Matter"
        else:
            seen_chapter = True
            chap_seq += 1
            title_slug = _slugify(title or f"Chapter-{chap_seq}")
            slug = f"{chap_seq:02d}-{title_slug}"
            display_title = title or f"Chapter {chap_seq}"

        filename = f"{slug}.md"
        filepath = output_dir / filename
        filepath.write_text(f"# {display_title}\n\n{md}\n", encoding="utf-8")
        print(f"  wrote  {filename}  ({len(md):,} chars)")

        written.append({"filename": filename, "title": display_title, "text": md})

    return written


# ─── Character extraction ──────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "he","she","it","they","we","you","i","me","him","her","us","them",
    "his","its","their","our","your","my","this","that","these","those",
    "what","which","who","how","when","where","why","if","then","than",
    "not","no","so","as","be","is","was","are","were","been","have","has",
    "had","do","did","does","will","would","could","should","may","might",
    "shall","can","just","there","here","said","says","say","one","two",
    "chapter","part","book","page","end","new","old","first","last","next",
    "time","day","night","year","way","back","up","down","out","over","now",
    "still","even","also","only","very","more","most","some","any","each",
    "after","before","into","from","by","about","like","through","between",
}


def extract_character_names(chapters: list) -> list[tuple[str, int]]:
    """Return (name, mention_count) pairs for likely character names, sorted by frequency."""
    name_re = re.compile(
        r"\b(?:(?:Mr|Mrs|Ms|Dr|Prof|Sir|Lady|Lord|Captain|Sergeant|Officer)"
        r"\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    )

    raw: Counter = Counter()
    for chapter in chapters:
        text = html_to_markdown(chapter["html"])
        for m in name_re.finditer(text):
            candidate = m.group(0).strip()
            tokens = candidate.lower().split()
            if any(t in _STOPWORDS for t in tokens):
                continue
            if len(candidate) < 3:
                continue
            raw[candidate] += 1

    filtered = [(name, count) for name, count in raw.items() if count >= 2]
    filtered.sort(key=lambda x: -x[1])
    return filtered


def build_characters_md(metadata: dict, chapters: list) -> str:
    names = extract_character_names(chapters)

    lines = ["# Characters", ""]
    lines += [
        f"*Extracted from: {metadata.get('title','')}*  ",
        f"*Author: {metadata.get('creator','')}*",
        "",
        "> Character names were identified automatically by frequency analysis.",
        "> Review and fill in descriptions manually, or use Claude Code.",
        "",
    ]

    if not names:
        lines.append("_No recurring proper names detected._\n")
        return "\n".join(lines)

    for name, count in names:
        lines += [
            f"## {name}",
            "",
            f"- **Mentions**: {count}",
            f"- **Role**: ",
            f"- **Physical description**: ",
            f"- **Personality**: ",
            f"- **Background**: ",
            f"- **Arc**: ",
            f"- **Notable quotes**: ",
            "",
        ]

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    default_output = Path(__file__).parent / "chapters"

    parser = argparse.ArgumentParser(
        description="Convert an ePub to per-chapter Markdown files for Multivoice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("epub", help="Path to the .epub file")
    parser.add_argument("--output-dir", "-o", default=str(default_output),
        help=f"Output directory (default: {default_output})")
    args = parser.parse_args()

    epub_path = Path(args.epub).resolve()
    if not epub_path.exists():
        sys.exit(f"Error: file not found: {epub_path}")
    if epub_path.suffix.lower() != ".epub":
        print(f"Warning: expected a .epub file, got '{epub_path.suffix}'", file=sys.stderr)

    output_dir = Path(args.output_dir).resolve()

    print(f"\nParsing {epub_path.name} …")
    metadata, chapters = parse_epub(str(epub_path))
    print(f"  Title   : {metadata.get('title')  or '(unknown)'}")
    print(f"  Author  : {metadata.get('creator') or '(unknown)'}")
    print(f"  Spine items : {len(chapters)}")

    print(f"\nWriting chapters → {output_dir}/")
    written = write_chapters(metadata, chapters, output_dir)
    print(f"\n  {len(written)} chapters written.")

    print("\nExtracting character names …")
    char_md = build_characters_md(metadata, chapters)
    char_path = output_dir / "characters.md"
    char_path.write_text(char_md, encoding="utf-8")
    print(f"  wrote  characters.md")

    print(f"\nDone.  Output: {output_dir}/")
    print("\nNext steps:")
    print("  1. Review chapters/ and characters.md")
    print("  2. Use Claude Code to convert chapters to screenplay format")
    print("  3. Use Claude Code to generate voice_config.json")
    print("  4. Run: python tts_generator.py --chapter 1 --assemble")


if __name__ == "__main__":
    main()
