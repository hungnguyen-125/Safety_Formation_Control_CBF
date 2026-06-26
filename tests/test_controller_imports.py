from safety_formation.control_law import BaseController as OldBaseController
from safety_formation.control_law import CentralizedFormationControl as OldCentralizedFormation
from safety_formation.control_law import DistributedFormationControl as OldDistributedFormation
from safety_formation.control_law.cbf import CBFTopology as OldCBFTopology
from safety_formation.control_law.cbf import CentralizedCBF as OldCentralizedCBF
from safety_formation.control_law.cbf import DeadlockManager as OldDeadlockManager
from safety_formation.control_law.cbf import DecentralizedCBF as OldDecentralizedCBF
from safety_formation.control_law.cbf.base_cbf import BaseCBF as OldBaseCBF
from safety_formation.control_law.cbf.cbf_topology import CBFTopology as OldCBFTopologyModule
from safety_formation.control_law.cbf.centralized_cbf import CentralizedCBF as OldCentralizedCBFModule
from safety_formation.control_law.cbf.deadlock_manager import DeadlockManager as OldDeadlockManagerModule
from safety_formation.control_law.cbf.decentralized_cbf import DecentralizedCBF as OldDecentralizedCBFModule
from safety_formation.control_law.nominal import CentralizedFormationControl as OldCentralizedNominal
from safety_formation.control_law.nominal import DistributedFormationControl as OldDistributedNominal
from safety_formation.control_law.nominal.centralized_formation import (
    CentralizedFormationControl as OldCentralizedNominalModule,
)
from safety_formation.control_law.nominal.distributed_formation import (
    DistributedFormationControl as OldDistributedNominalModule,
)
from safety_formation.controllers import BaseController
from safety_formation.controllers import CentralizedFormationControl
from safety_formation.controllers import DistributedFormationControl
from safety_formation.controllers.cbf import BaseCBF
from safety_formation.controllers.cbf import CBFTopology
from safety_formation.controllers.cbf import CentralizedCBF
from safety_formation.controllers.cbf import DeadlockManager
from safety_formation.controllers.cbf import DecentralizedCBF


def test_control_law_imports_remain_compatible():
    assert OldBaseController is BaseController
    assert OldCentralizedFormation is CentralizedFormationControl
    assert OldDistributedFormation is DistributedFormationControl


def test_nominal_imports_remain_compatible():
    assert OldCentralizedNominal is CentralizedFormationControl
    assert OldDistributedNominal is DistributedFormationControl
    assert OldCentralizedNominalModule is CentralizedFormationControl
    assert OldDistributedNominalModule is DistributedFormationControl


def test_cbf_imports_remain_compatible():
    assert OldBaseCBF is BaseCBF
    assert OldCentralizedCBF is CentralizedCBF
    assert OldDecentralizedCBF is DecentralizedCBF
    assert OldCBFTopology is CBFTopology
    assert OldDeadlockManager is DeadlockManager
    assert OldCentralizedCBFModule is CentralizedCBF
    assert OldDecentralizedCBFModule is DecentralizedCBF
    assert OldCBFTopologyModule is CBFTopology
    assert OldDeadlockManagerModule is DeadlockManager
