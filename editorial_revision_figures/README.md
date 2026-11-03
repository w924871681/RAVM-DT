# Figures revised for editorial readability

These four PDFs are the manuscript figures revised for larger labels and legends.
The original g04 figures remain in their historical snapshot. The script reads the
existing g04 aggregate inputs from `g04_论文版本/输出结果/绘图输入` and writes its
figures to `editorial_revision_figures/output`; it does not train models or alter inputs.
Run `python editorial_revision_figures/redraw_readable.py` from a repository checkout
with Python, NumPy, Matplotlib and Pillow installed. Installed fonts may affect layout.
The script uses the already published E3 method summaries, not the raw Alibaba data.
