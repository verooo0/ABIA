class AzamonOperator(object):
    pass

class AssignPackage(AzamonOperator):
    def __init__(self, package_id: int, offer_id: int):
        self.package_id = package_id
        self.offer_id = offer_id

    def __repr__(self) -> str:
        return f"Assign package {self.package_id} to offer {self.offer_id}"

class SwapAssignments(AzamonOperator):
    def __init__(self, package_id_1: int, package_id_2: int):
        self.package_id_1 = package_id_1
        self.package_id_2 = package_id_2

    def __repr__(self) -> str:
        return f"Swap assignments of packages {self.package_id_1} and {self.package_id_2}"
    
class RemovePackage(AzamonOperator):
    def __init__(self, package_id: int, offer_id: int):
        self.package_id = package_id
        self.offer_id = offer_id  

    def __repr__(self) -> str:
        return f"Remove package {self.package_id} from offer {self.offer_id}"

class InsertPackage(AzamonOperator):
    def __init__(self, package_id: int, offer_id: int):
        self.package_id = package_id
        self.offer_id = offer_id

    def __repr__(self) -> str:
        return f"Insert package {self.package_id} into offer {self.offer_id}"