"""Offline multi-dataset comparison; export validity never becomes a likeness score."""
import csv,html
from pathlib import Path
from .io import write

def summarize(output,report):
 output=Path(output);write(output/'batch.report.json',report);rows=[]
 for dataset in report['datasets']:
  variants=dataset.get('variants') or [{'backend':'none','status':dataset['status']}]
  for v in variants:
   diagnostic=v.get('diagnostics') or {};row={'dataset':dataset['id'],'backend':v['backend'],'status':v['status'],'input_fingerprint':dataset.get('input_fingerprint'),'face_proxy_error':diagnostic.get('eye_contour_error_proxy_median'),'eye_semantic_pixels':sum(x.get('eye_semantic_pixels',0) for x in diagnostic.get('eyes',[])) if diagnostic else None,'seconds':v.get('seconds'),'figurine_status':v.get('figurine_status','NOT_RUN'),'hair_exported':v.get('hair_candidate_counts',{}).get('exported'),'hair_rejected':v.get('hair_candidate_counts',{}).get('rejected'),'error':v.get('error') or v.get('appearance_error') or v.get('figurine_error') or '; '.join(dataset.get('errors',[])),'ready_for_machine':False}
   rows.append(row)
 with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
  fields=['dataset','backend','status','input_fingerprint','face_proxy_error','eye_semantic_pixels','seconds','figurine_status','hair_exported','hair_rejected','error','ready_for_machine'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 esc=html.escape;text='<!doctype html><html lang="vi"><meta charset="utf-8"><title>HeadLab26 · Dataset benchmark</title><style>body{font:16px system-ui;background:#171c24;color:#eee;margin:24px;max-width:1600px}section{background:#252f3b;padding:18px;margin:18px 0;border-radius:12px}img{max-width:100%;height:auto}a{color:#9cd0ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}td,th{border:1px solid #536070;padding:8px;text-align:left}.notice{background:#493b28;padding:15px}</style><h1>HeadLab26 · so sánh bộ ảnh</h1><p class="notice">Status '+esc(report['status'])+'. Hình học hợp lệ, độ giống, màu và khả năng in được đánh giá riêng. Điểm mí mắt là proxy, không phải ground truth. Mọi nhánh chưa kiểm chứng máy in đều ready_for_machine=false.</p><p><a href="summary.csv">CSV</a> · <a href="batch.report.json">JSON đầy đủ</a></p><table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in ['Dataset','Backend','Mặt','Lid proxy ↓','Fit + mặt (giây)','Tóc','Xuất / loại'])+'</tr>'
 options=lambda values:''.join('<option value="'+esc(str(v))+'">'+esc(str(v))+'</option>' for v in values)
 controls='<p><label>Bộ ảnh <select id="case-filter"><option value="">Tất cả</option>'+options(d['id'] for d in report['datasets'])+'</select></label> · <label>Nhánh <select id="backend-filter"><option value="">Tất cả</option>'+options(sorted({r['backend'] for r in rows}))+'</select></label></p>'
 text=text.replace('<table>',controls+'<table>',1)
 for row in rows:text+='<tr data-case="'+esc(row['dataset'])+'" data-backend="'+esc(row['backend'])+'">'+''.join('<td>'+esc(str(row.get(x,'')))+'</td>' for x in ['dataset','backend','status','face_proxy_error','seconds','figurine_status'])+'<td>'+esc(str(row['hair_exported']))+' / '+esc(str(row['hair_rejected']))+'</td></tr>'
 text+='</table>'
 for d in report['datasets']:
  text+='<section data-case="'+esc(d['id'])+'"><h2>'+esc(d['id'])+' · '+esc(d['status'])+'</h2>'
  if d.get('errors'):text+='<pre>'+esc('\n'.join(d['errors']))+'</pre>'
  text+='<p>Input fingerprint: '+esc(d.get('input_fingerprint',''))+'</p>'
  for v in d.get('variants',[]):
   prefix='datasets/'+d['id']+'/'+v['backend'];text+='<article data-backend="'+esc(v['backend'])+'"><h3>'+esc(v['backend'])+' · '+esc(v['status'])+'</h3>'
   if v.get('figurine_status'):text+='<p>Tóc: '+esc(v['figurine_status'])+' · '+esc(str(v.get('hair_candidate_counts',{})))+'</p>'
   for field in ['error','appearance_error','figurine_error']:
    if v.get(field):text+='<pre>'+esc(field+': '+v[field])+'</pre>'
   for name in ['face_gray.png','face_source.png','face_canonical.png','eyes_closeup.png']:
    path=output/prefix/'diagnostics'/name
    if path.is_file():text+='<img loading="lazy" src="'+esc(prefix+'/diagnostics/'+name)+'">'
   for slot in ['front','left','right']:
    if (output/prefix/'diagnostics'/(slot+'.eye_overlay.png')).is_file():text+='<a href="'+esc(prefix+'/diagnostics/'+slot+'.eye_overlay.png')+'">'+slot+' overlay</a> · '
   if (output/prefix/'figurine/review.html').is_file():text+='<p><a href="'+esc(prefix+'/figurine/review.html')+'">Tóc, kính, geometry và support</a></p>'
   if (output/prefix/'report.json').is_file():text+='<p><a href="'+esc(prefix+'/report.json')+'">Báo cáo nhánh</a></p>'
   text+='</article>'
  if d.get('trust'):
   text+='<p>TRUST: '+esc(d['trust']['status'])+'</p>'
   for slot in ['front','left','right']:
    rel='datasets/'+d['id']+'/trust/'+slot+'.albedo.png'
    if (output/rel).is_file():text+='<a href="'+esc(rel)+'">'+slot+' UV albedo (model estimate)</a> · '
  text+='</section>'
 text+='<p>Chưa có thang điểm thẩm mỹ hoặc đánh giá người dùng độc lập; không dùng số case thành số người khảo sát.</p></html>'
 text=text.replace('</html>','<script>function filterCases(){const c=document.getElementById("case-filter").value,b=document.getElementById("backend-filter").value;for(const e of document.querySelectorAll("[data-case]")){e.hidden=!!((c&&e.dataset.case!==c)||(e.dataset.backend&&b&&e.dataset.backend!==b));}for(const e of document.querySelectorAll("article[data-backend]")){e.hidden=!!(b&&e.dataset.backend!==b);}}for(const id of ["case-filter","backend-filter"]){document.getElementById(id).addEventListener("change",filterCases);}</script></html>')
 (output/'index.html').write_text(text,encoding='utf-8')
