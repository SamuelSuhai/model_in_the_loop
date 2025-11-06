from typing import Any, Dict, List, Optional, Union

def get_rois_in_field_restriction_str(field_key: Dict[str, Any],roi_id_subset:Optional[List[int]] = None) -> Union[str, Dict]:
    """
    Constructs a restriction string for the given field_key and optional roi_id_subset.
    """
    if field_key == {}:
        return {} # no restriction

    complete_restriction = " AND ".join([f"{k}='{v}'" for k,v in field_key.items()])
    if roi_id_subset is not None:
        roi_restriction_string = f"roi_id in {str(tuple(map(int,roi_id_subset)))}" if len(roi_id_subset) >= 2 else f"roi_id={str(roi_id_subset[0])}"
        complete_restriction =  complete_restriction + " AND " + roi_restriction_string

    return complete_restriction

