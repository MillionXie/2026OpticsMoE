"""Human-directed three-class replacement; old data untouched, no optical test search."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import prepare_caltech_selected as original

HERE=Path(__file__).resolve().parent
EXCLUDED=['Faces','Motorbikes','airplanes']
KEEP=['Leopards','bonsai','crab','crayfish','grand_piano','helicopter','kangaroo','ketch','laptop','lotus','minaret','revolver','schooner','trilobite','umbrella','watch','yin_yang']


def replacement(train,validation,count=20):
    assert count==20 and set(KEEP)<=set(train)
    classes=sorted(c for c in train if c not in EXCLUDED);centroids={c:np.mean(train[c],axis=0) for c in classes};steps=[]
    while True:
        means=np.stack([centroids[c] for c in classes]);scores=[]
        for k,c in enumerate(classes):
            distances=((validation[c][:,None]-means[None])**2).sum(2);other=np.min(np.delete(distances,k,axis=1),axis=1)
            scores.append(dict(class_name=c,recall=float(np.mean(distances.argmin(1)==k)),margin=float(np.mean(other-distances[:,k])),validation_count=len(validation[c]),protected_previous_class=c in KEEP))
        steps.append(dict(classes=classes[:],macro_recall=float(np.mean([r['recall'] for r in scores])),scores=scores))
        if len(classes)==20:break
        worst=min((r for r in scores if r['class_name'] not in KEEP),key=lambda r:(r['recall'],r['margin'],r['class_name']));classes.remove(worst['class_name'])
    assert len(set(classes)-set(KEEP))==3 and not set(EXCLUDED)&set(classes)
    return classes,steps


if __name__=='__main__':
    # Reuse the original archive/grouping/role-pinning implementation, not a copied pipeline.
    out=Path(sys.argv[sys.argv.index('--out')+1]);original.choose_classes=replacement;original.main()
    selection=json.loads((out/'category_selection.json').read_text())
    selection.update(human_excluded_classes=EXCLUDED,preserved_classes=KEEP,replacements=[c for c in selection['selected_classes'] if c not in KEEP],
        exclusions_based_on_prior_development_results=True,replacement_screen_uses_test_pixels_or_scores=False)
    original.save(out/'category_selection.json',selection)
    m=json.loads((out/'data_manifest.json').read_text());m.update(category_selection_sha256=original.sha(out/'category_selection.json'),
        preparation_wrapper_sha256=original.sha(__file__),human_excluded_classes=EXCLUDED,
        replacements=selection['replacements'],prior_development_test_informed_exclusions=True,
        limitations='Human exclusions follow observed development results. Fixed train/val thumbnail screen picks only three replacements, all17 remaining classes preserved. This is a changed development task, not an independent held-out accuracy claim or a controlled comparison with old20.')
    original.save(out/'data_manifest.json',m)
    print(json.dumps(dict(classes=m['classes'],replacements=m['replacements'],supports=m['supports'],cache_sha256=m['cache_sha256'],manifest_sha256=original.sha(out/'data_manifest.json'))))
