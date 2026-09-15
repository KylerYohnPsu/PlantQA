"""A class to take plant record information and save as a json object"""
import re
import json
class plant_record:


    def __init__(self, json_data):
        self.symbol = json_data["Symbol"]
        self.scientific_name = self._stripHtml(json_data["ScientificName"])
        self.rank = json_data['Rank']
        self.group = json_data.get("Group")
        self.common_name = json_data.get("CommonName")
        self.other_common_names = json_data.get("OtherCommonNames") or []
        self.durations = json_data.get('Durations') or []
        self.growth_habits = json_data.get('GrowthHabits') or []
        self.fact_sheet_urls = json_data.get('FactSheetUrls') or []
        self.plant_guide_urls = json_data.get('PlantGuideUrls') or []
        self.native_regions = [{'region': n['Region'], 'status': n['Type']}
                               for n in (json_data.get("NativeStatuses") or [])]
        notes = json_data.get("PlantNotes") or {}
        self.plant_notes = {k: v for k, v in notes.items() if v not in (None, "", [])}
        self.Has_documents = bool(self.fact_sheet_urls or self.plant_guide_urls)
        ancestors = json_data.get("Ancestors") or []
        self.family = self._stripHtml(
            next((a['ScientificName'] for a in ancestors if a.get("Rank") == "Family"), None)
        )
        self.genus = self._stripHtml(
            next((a['ScientificName'] for a in ancestors if a.get("Rank") == "Genus"), None)
        )

    def to_dict(self):
        return self.__dict__
    
    @staticmethod
    def _stripHtml(sname):
        if not sname:
            return sname
        return re.sub("<[^>]+>", "", sname).strip()

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent = 2)