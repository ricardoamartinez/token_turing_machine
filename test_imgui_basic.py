"""
Basic test script for Dear ImGui.

This script tests that Dear ImGui is installed and working correctly.
"""

import sys
import imgui

def main():
    """Run the test."""
    try:
        # Create ImGui context
        imgui.create_context()

        # Print ImGui version
        version = imgui.get_version()
        print(f"ImGui version: {version}")

        # Set display size
        io = imgui.get_io()
        io.display_size = 1280, 720

        # Create a new frame
        imgui.new_frame()

        # Create a window
        imgui.begin("Test Window")
        imgui.text("Hello, world!")
        imgui.end()

        # Render
        imgui.render()

        print("Dear ImGui is working correctly!")
        print("Terminal Validation: ImGui is properly installed and functioning.")

        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
