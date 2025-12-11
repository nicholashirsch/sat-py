import pygfx as gfx
from rendercanvas.auto import RenderCanvas

class TempRenderEngine:
    def __init__(self):
        self.central_body = gfx.Mesh(
            gfx.sphere_geometry(radius=1, width_segments=64, height_segments=32),
            gfx.MeshPhongMaterial(color="#00CC55")
        )
        self.canvas = RenderCanvas(size=(200, 200), title="TBD")
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight())
        self.scene.add(gfx.DirectionalLight())
        self.scene.add(self.central_body)

    def render(self):
        gfx.show(self.scene, renderer=self.renderer)

    def create_frame(self):
        pass

    def event_handler(self, event):
        event_type = event.type
        key = event.key.lower()

        match event_type:
            case "key_down":
                pass
            case "key_up":
                pass
