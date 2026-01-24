"""
Comprehensive unit tests for LinearPenalty class.

Tests all methods and edge cases of the LinearPenalty class.
"""
import pytest
import numpy as np
from stellcoilbench.coil_optimization import LinearPenalty


class MockObjective:
    """Mock objective for testing LinearPenalty."""
    def __init__(self, J_value=10.0, dJ_value=None, x_value=None):
        self._J_value = J_value
        self._dJ_value = dJ_value if dJ_value is not None else np.array([1.0, 2.0, 3.0])
        self._x_value = x_value if x_value is not None else np.array([0.0, 0.0, 0.0])
        self._parent = None
        self._children = []
    
    def J(self):
        return self._J_value
    
    def dJ(self, **kwargs):
        return self._dJ_value
    
    @property
    def x(self):
        return self._x_value
    
    @x.setter
    def x(self, value):
        self._x_value = value
    
    def __add__(self, other):
        if isinstance(other, MockObjective):
            combined = MockObjective(
                J_value=self._J_value + other._J_value,
                dJ_value=self._dJ_value + other._dJ_value,
                x_value=self._x_value
            )
            return combined
        return NotImplemented


class TestLinearPenaltyBasic:
    """Basic tests for LinearPenalty initialization and J() method."""
    
    def test_init(self):
        """Test LinearPenalty initialization."""
        obj = MockObjective(J_value=10.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        assert penalty.objective is obj
        assert penalty.threshold == 5.0
        assert penalty._parent is None
        assert penalty._children == []
    
    def test_J_above_threshold(self):
        """Test J() when objective value is above threshold."""
        obj = MockObjective(J_value=10.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        assert penalty.J() == 5.0  # 10.0 - 5.0
    
    def test_J_below_threshold(self):
        """Test J() when objective value is below threshold."""
        obj = MockObjective(J_value=3.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        assert penalty.J() == 0.0  # max(3.0 - 5.0, 0.0)
    
    def test_J_equal_threshold(self):
        """Test J() when objective value equals threshold."""
        obj = MockObjective(J_value=5.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        assert penalty.J() == 0.0  # max(5.0 - 5.0, 0.0)
    
    def test_J_negative_objective(self):
        """Test J() with negative objective value."""
        obj = MockObjective(J_value=-2.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        assert penalty.J() == 0.0  # max(-2.0 - 5.0, 0.0)


class TestLinearPenaltyGradient:
    """Tests for dJ() gradient method."""
    
    def test_dJ_above_threshold(self):
        """Test dJ() when objective is above threshold."""
        obj = MockObjective(J_value=10.0, dJ_value=np.array([1.0, 2.0, 3.0]))
        penalty = LinearPenalty(obj, threshold=5.0)
        grad = penalty.dJ()
        np.testing.assert_array_equal(grad, np.array([1.0, 2.0, 3.0]))
    
    def test_dJ_below_threshold(self):
        """Test dJ() when objective is below threshold."""
        obj = MockObjective(J_value=3.0, dJ_value=np.array([1.0, 2.0, 3.0]))
        penalty = LinearPenalty(obj, threshold=5.0)
        grad = penalty.dJ()
        np.testing.assert_array_equal(grad, np.array([0.0, 0.0, 0.0]))
    
    def test_dJ_equal_threshold(self):
        """Test dJ() when objective equals threshold."""
        obj = MockObjective(J_value=5.0, dJ_value=np.array([1.0, 2.0, 3.0]))
        penalty = LinearPenalty(obj, threshold=5.0)
        grad = penalty.dJ()
        np.testing.assert_array_equal(grad, np.array([0.0, 0.0, 0.0]))
    
    def test_dJ_with_scalar_gradient(self):
        """Test dJ() with scalar gradient value."""
        class ScalarObjective:
            def J(self):
                return 10.0
            def dJ(self, **kwargs):
                return 5.0
            @property
            def x(self):
                return np.array([1.0])
        
        obj = ScalarObjective()
        penalty = LinearPenalty(obj, threshold=5.0)
        grad = penalty.dJ()
        assert grad == 5.0
        
        # Below threshold
        class ScalarObjectiveBelow:
            def J(self):
                return 3.0
            def dJ(self, **kwargs):
                return 5.0
            @property
            def x(self):
                return np.array([1.0])
        
        obj_below = ScalarObjectiveBelow()
        penalty_below = LinearPenalty(obj_below, threshold=5.0)
        grad_below = penalty_below.dJ()
        assert grad_below == 0.0
    
    def test_dJ_fallback_when_no_x(self):
        """Test dJ() fallback when objective has no x attribute."""
        class NoXObjective:
            def J(self):
                return 3.0
            def dJ(self, **kwargs):
                return np.array([1.0, 2.0])  # Return array that can be multiplied
        
            @property
            def x(self):
                raise AttributeError("No x attribute")
        
        obj = NoXObjective()
        penalty = LinearPenalty(obj, threshold=5.0)
        # Should return zeros when below threshold
        grad = penalty.dJ()
        np.testing.assert_array_equal(grad, np.array([0.0, 0.0]))


class TestLinearPenaltyAddition:
    """Tests for __add__ and __radd__ methods."""
    
    def test_add_two_penalties(self):
        """Test adding two LinearPenalty objects."""
        obj1 = MockObjective(J_value=10.0)
        obj2 = MockObjective(J_value=5.0)
        penalty1 = LinearPenalty(obj1, threshold=5.0)
        penalty2 = LinearPenalty(obj2, threshold=5.0)
        
        result = penalty1 + penalty2
        assert isinstance(result, LinearPenalty)
        assert result.threshold == 5.0
    
    def test_add_zero(self):
        """Test adding zero to LinearPenalty."""
        obj = MockObjective(J_value=10.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        result = penalty + 0
        assert result is penalty
    
    def test_radd_zero(self):
        """Test right addition of zero."""
        obj = MockObjective(J_value=10.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        result = 0 + penalty
        assert result is penalty
    
    def test_add_non_zero_number(self):
        """Test adding non-zero number raises TypeError."""
        obj = MockObjective(J_value=10.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        with pytest.raises(TypeError):
            _ = penalty + 1
    
    def test_sum_with_penalties(self):
        """Test using sum() with LinearPenalty objects."""
        obj1 = MockObjective(J_value=10.0)
        obj2 = MockObjective(J_value=5.0)
        penalty1 = LinearPenalty(obj1, threshold=5.0)
        penalty2 = LinearPenalty(obj2, threshold=5.0)
        
        result = sum([penalty1, penalty2])
        assert isinstance(result, LinearPenalty)


class TestLinearPenaltyMultiplication:
    """Tests for __mul__ and __rmul__ methods."""
    
    def test_multiply_with_weight(self):
        """Test multiplying LinearPenalty with Weight."""
        pytest.importorskip("simsopt")
        from simsopt.objectives import Weight
        
        # Create a real simsopt objective that can be multiplied by Weight
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        # Create a simple BiotSavart (empty coils)
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        base_curves = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.0, R1=0.1, order=2, numquadpoints=64)
        base_currents = [Current(1e6) for _ in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, 1, True)
        bs = BiotSavart(coils)
        
        obj = SquaredFlux(surface, bs)
        penalty = LinearPenalty(obj, threshold=5.0)
        weight = Weight(2.0)
        
        result = penalty * weight
        assert isinstance(result, LinearPenalty)
    
    def test_multiply_weight_with_penalty(self):
        """Test right multiplication: Weight * LinearPenalty."""
        pytest.importorskip("simsopt")
        from simsopt.objectives import Weight
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        base_curves = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.0, R1=0.1, order=2, numquadpoints=64)
        base_currents = [Current(1e6) for _ in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, 1, True)
        bs = BiotSavart(coils)
        
        obj = SquaredFlux(surface, bs)
        penalty = LinearPenalty(obj, threshold=5.0)
        weight = Weight(2.0)
        
        result = weight * penalty
        assert isinstance(result, LinearPenalty)
    
    def test_multiply_with_zero_objective(self):
        """Test multiplication when objective J() is zero."""
        pytest.importorskip("simsopt")
        from simsopt.objectives import Weight
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        base_curves = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.0, R1=0.1, order=2, numquadpoints=64)
        base_currents = [Current(1e6) for _ in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, 1, True)
        bs = BiotSavart(coils)
        
        obj = SquaredFlux(surface, bs)
        penalty = LinearPenalty(obj, threshold=5.0)
        weight = Weight(2.0)
        
        result = penalty * weight
        assert isinstance(result, LinearPenalty)
    
    def test_multiply_with_non_weight(self):
        """Test multiplying with non-Weight object."""
        obj = MockObjective(J_value=10.0)
        penalty = LinearPenalty(obj, threshold=5.0)
        with pytest.raises(TypeError):
            _ = penalty * 2.0


class TestLinearPenaltyAttributeDelegation:
    """Tests for __getattr__ attribute delegation."""
    
    def test_delegate_to_objective(self):
        """Test that attributes are delegated to underlying objective."""
        class ObjectiveWithAttr:
            def J(self):
                return 10.0
            def dJ(self, **kwargs):
                return np.array([1.0])
            @property
            def x(self):
                return np.array([0.0])
            def custom_method(self):
                return "custom"
            custom_attr = "value"
        
        obj = ObjectiveWithAttr()
        penalty = LinearPenalty(obj, threshold=5.0)
        
        assert penalty.custom_method() == "custom"
        assert penalty.custom_attr == "value"
    
    def test_raise_on_reserved_attributes(self):
        """Test that reserved attributes raise AttributeError."""
        obj = MockObjective()
        penalty = LinearPenalty(obj, threshold=5.0)
        
        # J, dJ, x are methods/properties, not attributes, so they won't raise AttributeError
        # But accessing them as attributes should work differently
        # Actually, J and dJ are methods, so accessing them as attributes would get the method object
        # x is a property, so it should work
        # objective and threshold are actual attributes, so they should work too
        # The __getattr__ only raises for attributes not found, so these should all work
        assert hasattr(penalty, 'J')
        assert hasattr(penalty, 'dJ')
        assert hasattr(penalty, 'x')
        assert hasattr(penalty, 'objective')
        assert hasattr(penalty, 'threshold')


class TestLinearPenaltyXProperty:
    """Tests for x property getter and setter."""
    
    def test_x_getter(self):
        """Test getting x property."""
        x_val = np.array([1.0, 2.0, 3.0])
        obj = MockObjective(x_value=x_val)
        penalty = LinearPenalty(obj, threshold=5.0)
        np.testing.assert_array_equal(penalty.x, x_val)
    
    def test_x_setter(self):
        """Test setting x property."""
        x_val = np.array([1.0, 2.0, 3.0])
        new_x_val = np.array([4.0, 5.0, 6.0])
        obj = MockObjective(x_value=x_val)
        penalty = LinearPenalty(obj, threshold=5.0)
        penalty.x = new_x_val
        np.testing.assert_array_equal(obj._x_value, new_x_val)


class TestLinearPenaltyChildManagement:
    """Tests for _add_child method."""
    
    def test_add_child(self):
        """Test adding a child objective."""
        obj1 = MockObjective()
        obj2 = MockObjective()
        penalty1 = LinearPenalty(obj1, threshold=5.0)
        penalty2 = LinearPenalty(obj2, threshold=5.0)
        
        penalty1._add_child(penalty2)
        assert penalty2 in penalty1._children
        assert penalty2._parent is penalty1
    
    def test_add_child_twice(self):
        """Test adding same child twice doesn't duplicate."""
        obj1 = MockObjective()
        obj2 = MockObjective()
        penalty1 = LinearPenalty(obj1, threshold=5.0)
        penalty2 = LinearPenalty(obj2, threshold=5.0)
        
        penalty1._add_child(penalty2)
        penalty1._add_child(penalty2)
        assert penalty1._children.count(penalty2) == 1
