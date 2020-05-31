from .dataset import Dataset


def Wine():
    db = Dataset()
    db.get_data()
    db.prepare_data()
    db.graphs()
    db.data_description()
    db.correlation_color()
    db.correlation_quality()
    db.quality_alcohol_relation()
    db.quality_acidity_relation()
    db.type_alcohol_quality_acidity()
    db.classifier()
