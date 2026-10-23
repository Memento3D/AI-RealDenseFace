"""Create an unlit photo-color viewer without altering saved colors or geometry."""
import argparse
from pathlib import Path
from uuid import uuid4


def patch_html(html):
    changes = {
        'uniform bool uHeat;': 'uniform bool uPhotoColor;uniform bool uHeat;',
        'gl_FragColor=vec4(color*light,1.);': 'gl_FragColor=vec4(uPhotoColor && !uHeat ? color : color*light,1.);',
        "'Aspect','Heat'": "'Aspect','PhotoColor','Heat'",
        "const m=models[i];": "gl.uniform1i(U.PhotoColor,i===1?1:0);const m=models[i];",
        '<title>Memento3D · So sánh hình dáng</title>': '<title>Memento3D · Photo color without extra lighting</title>',
        'So sánh hình dáng 3D': 'Geometry / Original photo colors',
        'Hai khung luôn dùng chung góc nhìn, tỷ lệ và ánh sáng.': 'Hai khung dùng chung góc nhìn và tỷ lệ. Bản màu không chiếu sáng thêm.',
        'Cùng tâm và tỷ lệ với bản thử nghiệm': 'Geometry with studio shading',
        'Thay đổi hình học thực tế, không phóng đại': 'Photo RGB without extra shading; same geometry',
        ' Hiện lưới': ' Show wireframe',
        ' Màu độ dịch chuyển': ' Displacement (zero: same geometry)',
    }
    for old, new in changes.items():
        if old not in html:
            raise ValueError('Viewer format mismatch; no output written: '+old[:80])
        html = html.replace(old,new)
    return html


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    folders=sorted(args.run_dir.glob('photo_color_*/head.photo_color.html'),key=lambda f:f.stat().st_mtime,reverse=True)
    if not folders:
        raise SystemExit('No photo_color_*/head.photo_color.html found inside --run-dir')
    source=folders[0]
    html=patch_html(source.read_text(encoding='utf-8'))
    output=source.with_name('head.photo_color.unlit_'+uuid4().hex[:8]+'.html')
    with output.open('x',encoding='utf-8') as f:f.write(html)
    print('Original preserved: '+str(source.resolve()))
    print('Open: '+str(output.resolve()))


if __name__=='__main__':main()
