from safety_formation.controllers import (
    BaseController,
    CentralizedFormationControl,
    DistributedFormationControl,
)
from safety_formation.controllers.cbf import (
    BaseCBF,
    CBFTopology,
    CentralizedCBF,
    DeadlockManager,
    DecentralizedCBF,
)
from safety_formation.controllers.cbf.centralized import CentralizedCBF as CentralizedCBFModule
from safety_formation.controllers.cbf.decentralized import DecentralizedCBF as DecentralizedCBFModule
from safety_formation.controllers.nominal import (
    CentralizedFormationControl as CentralizedNominal,
)
from safety_formation.controllers.nominal import (
    DistributedFormationControl as DistributedNominal,
)


def test_controller_public_imports_are_available():
    assert BaseController.__name__ == "BaseController"
    assert CentralizedFormationControl is CentralizedNominal
    assert DistributedFormationControl is DistributedNominal


def test_cbf_public_imports_are_available():
    assert BaseCBF.__name__ == "BaseCBF"
    assert CentralizedCBF is CentralizedCBFModule
    assert DecentralizedCBF is DecentralizedCBFModule
    assert CBFTopology.__name__ == "CBFTopology"
    assert DeadlockManager.__name__ == "DeadlockManager"
