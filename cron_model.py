"""
Represents the Antikythera mechanism's cycles using cron expressions.
Each celestial body's motion is mapped to the smallest possible cron increment.
"""

from typing import Dict, List
import math

class CronModel:
    """Models celestial cycles using cron expressions."""
    
    def __init__(self, config_path: str):
        """Initialize the cron model with configuration.
        
        Args:
            config_path: Path to model configuration file
        """
        import json
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Base cron unit (1 minute)
        self.base_unit = 1
        
    def _period_to_cron(self, period: float) -> str:
        """Convert orbital period to cron expression.
        
        Args:
            period: Orbital period in Earth days
            
        Returns:
            Cron expression representing the cycle
        """
        # Convert period to minutes (smallest cron unit)
        minutes = period * 24 * 60
        
        if minutes < 1:
            return "* * * * * echo(*)"  # Sub-minute cycles
        
        # Round to nearest minute
        minutes = round(minutes)
        
        # Create cron pattern
        if minutes < 60:
            return f"*/{minutes} * * * * echo(*)"
        elif minutes < 1440:  # Less than a day
            hours = minutes // 60
            return f"0 */{hours} * * * echo(*)"
        else:  # Days or longer
            days = minutes // 1440
            return f"0 0 */{days} * * echo(*)"
            
    def get_cron_expressions(self) -> Dict[str, str]:
        """Get cron expressions for all celestial bodies.
        
        Returns:
            Dictionary mapping body names to cron expressions
        """
        expressions = {}
        for body_name, body_config in self.config['celestial_bodies'].items():
            period = body_config['orbital_period']
            expressions[body_name] = self._period_to_cron(period)
        return expressions
    
    def get_resonance_patterns(self) -> List[str]:
        """Get cron patterns representing gear resonances.
        
        Returns:
            List of cron expressions for resonance points
        """
        patterns = []
        bodies = list(self.config['celestial_bodies'].items())
        
        for i, (body1_name, body1_config) in enumerate(bodies):
            for body2_name, body2_config in bodies[i+1:]:
                # Find resonance period
                period1 = body1_config['orbital_period']
                period2 = body2_config['orbital_period']
                
                # Calculate resonance using gear ratios
                ratio1 = body1_config['gear_ratio']
                ratio2 = body2_config['gear_ratio']
                
                resonance_period = (period1 * period2) / math.gcd(
                    round(period1 * ratio2),
                    round(period2 * ratio1)
                )
                
                patterns.append({
                    'bodies': (body1_name, body2_name),
                    'cron': self._period_to_cron(resonance_period)
                })
        
        return patterns
    
    def __str__(self) -> str:
        """Get string representation of the cron model.
        
        Returns:
            Formatted string showing all cron expressions
        """
        expressions = self.get_cron_expressions()
        resonances = self.get_resonance_patterns()
        
        output = ["Celestial Body Cycles:"]
        for body, cron in expressions.items():
            output.append(f"  {body}: {cron}")
            
        output.append("\nResonance Patterns:")
        for pattern in resonances:
            bodies = pattern['bodies']
            output.append(
                f"  {bodies[0]}-{bodies[1]} resonance: {pattern['cron']}"
            )
            
        return "\n".join(output)

if __name__ == "__main__":
    # Example usage
    model = CronModel("config/model_config.json")
    print(model)
    
    # Output will look like:
    # Celestial Body Cycles:
    #   sun: * * * * * echo(*)
    #   moon: 0 */18 * * * echo(*)
    #   mars: 0 0 */687 * * * echo(*)
    #
    # Resonance Patterns:
    #   sun-moon resonance: 0 */6 * * * echo(*)
    #   sun-mars resonance: 0 0 */687 * * * echo(*)
    #   moon-mars resonance: 0 0 */19 * * * echo(*)