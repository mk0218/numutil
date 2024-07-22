use std::cmp::Ordering;

use super::{Integer, Sign};

impl Integer {
    fn cmp_abs(self: &Self, other: &Self) -> Option<Ordering> {
        use Ordering as Ord;

        let n1 = &self.value;
        let n2 = &other.value;

        let result = if n1.len() > n2.len() {
            Ord::Greater
        } else if n1.len() < n2.len() {
            Ord::Less
        } else {
            let it_1 = n1.iter().rev();
            let it_2 = n2.iter().rev();
            
            it_1.zip(it_2)
                .find_map(|(d1, d2)| {
                    if d1 > d2 {
                        Some(Ord::Greater)
                    } else if d1 < d2 {
                        Some(Ord::Less)
                    } else {
                        None
                    }
                })
                .map_or(Ord::Equal, |res| res)
        };

        Some(result)
    }
}

impl PartialEq for Integer {
    fn eq(&self, other: &Self) -> bool {
        let (v1, v2) = (&self.value, &other.value);
        let (s1, s2) = (self.sign, other.sign);

        if v1.len() == 0 && v2.len() == 0 {
            true
        } else if s1 != s2 {
            false
        } else {
            v1 == v2
        }
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Ordering as Ord;
        
        match (self.sign, other.sign) {
            (Sign::PLUS, Sign::MINUS) => Some(Ord::Greater),
            (Sign::MINUS, Sign::PLUS) => Some(Ord::Less),
            (Sign::PLUS, Sign::PLUS) => self.cmp_abs(other),
            (Sign::MINUS, Sign::MINUS) => other.cmp_abs(self)
        }
    }
}

impl Ord for Integer {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect(
            "partial_cmp for Integer must not return None."
        )
    }
}

#[cfg(test)]
mod test_integer_cmp {
    use super::*;

    #[test]
    fn zero() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![] };

        assert!(v1 == v2);
    }
    
    #[test]
    fn plus_minus() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123, 233] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![123, 233] };

        assert!(v1 > v2);
    }

    #[test]
    fn minus_plus() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![1, 12, 11] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![1, 12, 11] };

        assert!(v1 < v2);
    }

    #[test]
    fn plus_greater() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123, 222] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![123, 22] };

        assert!(v1 > v2);
    }

    #[test]
    fn plus_less() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123, 23] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![123, 232] };

        assert!(v1 < v2);
    }

    #[test]
    fn minus_greater() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![123, 123] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![124, 123] };

        assert!(v1 > v2);
    }

    #[test]
    fn minus_less() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![124, 123] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![123, 123] };

        assert!(v1 < v2);
    }
}
