from controller import Supervisor
import math
import time

class CameraController(Supervisor):
    def __init__(self):
        super().__init__()
        
        # Get the time step of the current world
        self.timestep = int(self.getBasicTimeStep())
        
        # Get handle to the viewpoint
        self.viewpoint = self.getFromDef("viewpoint")
        if not self.viewpoint:
            print("ERROR: Viewpoint not found! Make sure it has DEF name 'viewpoint'")
        else:
            print("Viewpoint found successfully")
            
            # Store initial viewpoint settings
            self.initial_position = self.viewpoint.getField("position").getSFVec3f()
            print("Initial camera position:", self.initial_position)
            
            self.initial_orientation = self.viewpoint.getField("orientation").getSFRotation()
            print("Initial camera orientation:", self.initial_orientation)
            
            # Try to get field of view
            try:
                self.initial_fov = self.viewpoint.getField("fieldOfView").getSFFloat()
                print("Initial field of view:", self.initial_fov)
            except Exception as e:
                print("Note: Could not get fieldOfView:", str(e))
                # Default to 45 degrees in radians if not available
                self.initial_fov = 0.785
                try:
                    # Try to add field of view if it doesn't exist
                    self.viewpoint.getField("fieldOfView").setSFFloat(self.initial_fov)
                except:
                    print("Could not set fieldOfView - zoom functions may not work")
        
        # Get handle to the robot node
        self.robot_node = self.getFromDef("MyBot")
        if not self.robot_node:
            print("ERROR: Robot node not found! Make sure it has DEF name 'MyBot'")
        else:
            print("Robot node found successfully")
    
    def zoom_in(self, duration=5.0, target_fov=0.3):
        """Gradually zoom in by reducing the field of view"""
        print("Starting simple zoom in...")
        try:
            start_time = self.getTime()
            start_fov = self.viewpoint.getField("fieldOfView").getSFFloat()
            print("Starting FOV:", start_fov, "Target FOV:", target_fov)
            
            while self.getTime() - start_time < duration:
                # Calculate interpolation factor (0 to 1)
                t = (self.getTime() - start_time) / duration
                
                # Calculate new FOV using linear interpolation
                current_fov = start_fov + t * (target_fov - start_fov)
                
                # Update the field of view
                self.viewpoint.getField("fieldOfView").setSFFloat(current_fov)
                
                # Step the simulation
                if self.step(self.timestep) == -1:
                    break
        except Exception as e:
            print("Error in zoom_in:", str(e))
    
    def zoom_out(self, duration=5.0):
        """Gradually zoom out to the initial field of view"""
        print("Starting zoom out...")
        try:
            start_time = self.getTime()
            start_fov = self.viewpoint.getField("fieldOfView").getSFFloat()
            print("Starting FOV:", start_fov, "Target FOV:", self.initial_fov)
            
            while self.getTime() - start_time < duration:
                # Calculate interpolation factor (0 to 1)
                t = (self.getTime() - start_time) / duration
                
                # Calculate new FOV using linear interpolation
                current_fov = start_fov + t * (self.initial_fov - start_fov)
                
                # Update the field of view
                self.viewpoint.getField("fieldOfView").setSFFloat(current_fov)
                
                # Step the simulation
                if self.step(self.timestep) == -1:
                    break
        except Exception as e:
            print("Error in zoom_out:", str(e))

    def move_camera(self, target_position, target_orientation, duration=5.0):
        """
        Smoothly move the camera to a new position and orientation over a specified duration.
        
        Args:
            target_position (list): Target [x, y, z] position for the camera
            target_orientation (list): Target [x, y, z, angle] orientation for the camera
            duration (float): Duration of the transition in seconds
        """
        print(f"Starting camera movement to position: {target_position}, orientation: {target_orientation}")
        try:
            start_time = self.getTime()
            
            # Get current position and orientation
            start_position = self.viewpoint.getField("position").getSFVec3f()
            start_orientation = self.viewpoint.getField("orientation").getSFRotation()
            
            print(f"Starting from position: {start_position}, orientation: {start_orientation}")
            
            while self.getTime() - start_time < duration:
                # Calculate interpolation factor (0 to 1)
                t = (self.getTime() - start_time) / duration
                
                # Apply easing function for smoother start/end (optional)
                # This uses a simple cubic easing: t = t³ (smooth start) + (1-t)³ (smooth end)
                t_eased = 3 * (t ** 2) - 2 * (t ** 3)
                
                # Calculate new position using linear interpolation
                current_position = [
                    start_position[0] + t_eased * (target_position[0] - start_position[0]),
                    start_position[1] + t_eased * (target_position[1] - start_position[1]),
                    start_position[2] + t_eased * (target_position[2] - start_position[2])
                ]
                
                # Calculate new orientation using spherical linear interpolation
                # For simplicity, we'll use linear interpolation here
                # Note: For more complex rotations, quaternion slerp would be better
                current_orientation = [
                    start_orientation[0] + t_eased * (target_orientation[0] - start_orientation[0]),
                    start_orientation[1] + t_eased * (target_orientation[1] - start_orientation[1]),
                    start_orientation[2] + t_eased * (target_orientation[2] - start_orientation[2]),
                    start_orientation[3] + t_eased * (target_orientation[3] - start_orientation[3])
                ]
                
                # Normalize the rotation axis vector
                axis_length = math.sqrt(current_orientation[0]**2 + 
                                        current_orientation[1]**2 + 
                                        current_orientation[2]**2)
                if axis_length > 0:
                    current_orientation[0] /= axis_length
                    current_orientation[1] /= axis_length
                    current_orientation[2] /= axis_length
                
                # Update the viewpoint
                self.viewpoint.getField("position").setSFVec3f(current_position)
                self.viewpoint.getField("orientation").setSFRotation(current_orientation)
                
                # Step the simulation
                if self.step(self.timestep) == -1:
                    break
            
            # Ensure we reach exactly the target values at the end
            self.viewpoint.getField("position").setSFVec3f(target_position)
            self.viewpoint.getField("orientation").setSFRotation(target_orientation)
            
            print(f"Camera movement completed. Position: {target_position}, Orientation: {target_orientation}")
        
        except Exception as e:
            print(f"Error in move_camera: {str(e)}")
    
    def run_camera_sequence(self):
        """Run the full camera sequence: zoom in, move to other side, follow, zoom out"""
        print("====== Starting camera sequence ======")
        
        # Wait a moment before starting to let the simulation stabilize
        for i in range(20):
            if self.step(self.timestep) == -1:
                return
        
        # Simple zoom in (150秒)
        print("====== Zooming in ======")
        self.zoom_in(duration=150.0, target_fov=0.3)

        # Example in the run_camera_sequence method
        # Move to a new viewpoint (100秒)
        new_position = [-11, -0.8, 10.6]  # [x, y, z]
        new_orientation = [-0.3, 0.8, 0.2, 0.8]  # [x, y, z, angle] - looking along Y axis
        self.move_camera(new_position, new_orientation, duration=100.0)

       # Wait at the new position (12.5秒)
        print("====== Pausing at new position ======")
        start_time = self.getTime()
        while self.getTime() - start_time < 12.5:
            if self.step(self.timestep) == -1:
                return
            
        # Move to a new viewpoint (150秒)
        new_position = [-4.04, -12.8, 18.8]  # [x, y, z]
        new_orientation = [-0.465, 0.393, 0.794, 1.67]  # [x, y, z, angle] - looking along Y axis
        self.move_camera(new_position, new_orientation, duration=150.0)
        
        # Zoom out (200秒)
        print("====== Zooming out ======")
        self.zoom_out(duration=200.0)
        
        print("====== Camera sequence completed ======")


# Main code
if __name__ == "__main__":
    print("Starting Camera Controller")
    controller = CameraController()
    controller.run_camera_sequence()
    
    # Continue running to keep the supervisor alive
    print("Camera sequence finished, keeping controller alive")
    while controller.step(controller.timestep) != -1:
        pass