//! 类型安全的索引定义

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct CellId(pub usize);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct FaceId(pub usize);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct NodeId(pub usize);

pub const INVALID_CELL: usize = usize::MAX;

macro_rules! impl_index_type {
    ($ty:ty, $name:expr) => {
        impl $ty {
            #[inline(always)]
            pub const fn new(v: usize) -> Self {
                Self(v)
            }

            #[inline(always)]
            pub const fn idx(self) -> usize {
                self.0
            }

            #[inline(always)]
            pub const fn is_valid(self) -> bool {
                self.0 != INVALID_CELL
            }
        }

        impl From<usize> for $ty {
            #[inline(always)]
            fn from(v: usize) -> Self {
                Self(v)
            }
        }

        impl From<$ty> for usize {
            #[inline(always)]
            fn from(v: $ty) -> Self {
                v.0
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", $name, self.0)
            }
        }

        impl std::ops::Add<usize> for $ty {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: usize) -> Self {
                Self(self.0.checked_add(rhs).expect("索引溢出"))
            }
        }

        impl std::ops::Sub<usize> for $ty {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: usize) -> Self {
                Self(self.0.checked_sub(rhs).expect("索引下溢"))
            }
        }

        impl std::ops::AddAssign<usize> for $ty {
            #[inline(always)]
            fn add_assign(&mut self, rhs: usize) {
                self.0 = self.0.checked_add(rhs).expect("索引溢出");
            }
        }

        impl std::ops::SubAssign<usize> for $ty {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: usize) {
                self.0 = self.0.checked_sub(rhs).expect("索引下溢");
            }
        }
    };
}

impl_index_type!(CellId, "Cell");
impl_index_type!(FaceId, "Face");
impl_index_type!(NodeId, "Node");

impl<T> std::ops::Index<CellId> for Vec<T> {
    type Output = T;
    fn index(&self, idx: CellId) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T> std::ops::IndexMut<CellId> for Vec<T> {
    fn index_mut(&mut self, idx: CellId) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

impl<T> std::ops::Index<FaceId> for Vec<T> {
    type Output = T;
    fn index(&self, idx: FaceId) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T> std::ops::IndexMut<FaceId> for Vec<T> {
    fn index_mut(&mut self, idx: FaceId) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

impl<T> std::ops::Index<NodeId> for Vec<T> {
    type Output = T;
    fn index(&self, idx: NodeId) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T> std::ops::IndexMut<NodeId> for Vec<T> {
    fn index_mut(&mut self, idx: NodeId) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

impl<T> std::ops::Index<CellId> for [T] {
    type Output = T;
    fn index(&self, idx: CellId) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T> std::ops::IndexMut<CellId> for [T] {
    fn index_mut(&mut self, idx: CellId) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

impl<T> std::ops::Index<FaceId> for [T] {
    type Output = T;
    fn index(&self, idx: FaceId) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T> std::ops::IndexMut<FaceId> for [T] {
    fn index_mut(&mut self, idx: FaceId) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

impl<T> std::ops::Index<NodeId> for [T] {
    type Output = T;
    fn index(&self, idx: NodeId) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T> std::ops::IndexMut<NodeId> for [T] {
    fn index_mut(&mut self, idx: NodeId) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_conversion() {
        let cell = CellId::new(42);
        assert_eq!(cell.idx(), 42);
        assert_eq!(usize::from(cell), 42);
        assert_eq!(CellId::from(42), cell);
    }

    #[test]
    fn test_invalid_cell() {
        let invalid = CellId::new(INVALID_CELL);
        assert!(!invalid.is_valid());

        let valid = CellId::new(0);
        assert!(valid.is_valid());
    }

    #[test]
    fn test_arithmetic() {
        let cell = CellId::new(10);
        assert_eq!((cell + 5).idx(), 15);
        assert_eq!((cell - 3).idx(), 7);
    }

    #[test]
    #[should_panic(expected = "索引下溢")]
    fn test_underflow() {
        let cell = CellId::new(5);
        let _ = cell - 10;
    }

    #[test]
    fn test_vec_index() {
        let data = vec![1.0, 2.0, 3.0];
        let idx = CellId::new(1);
        assert_eq!(data[idx], 2.0);
    }

    #[test]
    fn test_ordering() {
        let a = CellId::new(1);
        let b = CellId::new(2);
        assert!(a < b);

        let mut ids = vec![CellId::new(3), CellId::new(1), CellId::new(2)];
        ids.sort();
        assert_eq!(ids, vec![CellId::new(1), CellId::new(2), CellId::new(3)]);
    }
}
