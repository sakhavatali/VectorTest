#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use image::{ImageFormat, Rgba, RgbaImage};
use serde_json::{Value, json};
use vectorcraft_doc::{ImageBlob, ImageObject, Node, NodeId, NodeKind};
use vectorcraft_engine::Session;
use vectorcraft_geom::Affine;
use vectorcraft_testkit::{fixtures::Fixture, invariants::check_all};

fn test_image_png() -> Vec<u8> {
    let image = RgbaImage::from_fn(32, 32, |x, y| {
        if x < 4 || y < 4 || x >= 28 || y >= 28 {
            Rgba([0, 0, 0, 0])
        } else if x < 16 {
            Rgba([230, 50, 40, 255])
        } else {
            Rgba([20, 100, 230, 255])
        }
    });

    let mut bytes = Vec::new();
    image.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png).expect("encode test PNG");

    bytes
}

fn session_with_image(bytes: Vec<u8>) -> (Session, NodeId) {
    let mut session = Fixture::Empty.session();

    let image_id = session
        .edit("Place test image", |doc, selection| {
            doc.images.insert("low-poly-test".into(), ImageBlob::png(bytes));

            let id = doc.alloc_id();
            let image =
                ImageObject { key: "low-poly-test".into(), width: 32, height: 32, xf: Affine::IDENTITY, link: None, placement: Default::default() };

            doc.insert(doc.default_layer(), 0, Node::new(id, NodeKind::Image(Box::new(image))))?;

            selection.set([id]);
            Ok(id)
        })
        .expect("insert test image");

    (session, image_id)
}

#[test]
fn low_poly_command_handles_generation_and_boundary_options() {
    // Expected source-image visibility after generation.
    let cases: [(Value, Option<bool>); 5] = [
        (json!({}), Some(true)),
        (json!({"mode": "silhouette", "boundary": "preserve"}), Some(false)),
        (
            json!({
                "mode": "illustration",
                "colorMode": "vibrant",
                "boundary": "transparent"
            }),
            Some(false),
        ),
        (
            json!({
                "mode": "natural",
                "colorMode": "palette",
                "boundary": "whole"
            }),
            Some(true),
        ),
        (json!({"deleteRaster": true}), None),
    ];

    for (params, expected_visibility) in cases {
        let (mut session, image_id) = session_with_image(test_image_png());

        let result = session.execute("object.createLowPoly", &params).unwrap_or_else(|error| {
            panic!("Low-Poly failed for {params}: {error}");
        });

        let group_id = NodeId(result["id"].as_u64().expect("generated group ID"));

        let triangle_count = result["triangles"].as_u64().expect("triangle count");

        assert!(triangle_count > 0, "expected visible triangles for {params}");

        let state = session.doc().expect("document after generation");
        let group = state.doc.node(group_id).expect("generated group exists");

        assert_eq!(group.name.as_deref(), Some("Low-Poly Artwork"));

        match &group.kind {
            NodeKind::Group { children, .. } => {
                assert_eq!(children.len() as u64, triangle_count);
            }
            other => panic!("expected generated group, got {other:?}"),
        }

        assert_eq!(state.selection.objects.as_slice(), &[group_id], "generated artwork should be selected");

        match expected_visibility {
            Some(visible) => {
                let source = state.doc.node(image_id).expect("source image should remain");

                assert_eq!(source.visible, visible, "unexpected source visibility for {params}");
            }
            None => {
                assert!(state.doc.node(image_id).is_none(), "deleteRaster should remove the source image");
            }
        }

        check_all(&mut session).unwrap_or_else(|error| panic!("document invariant failed: {error}"));
    }
}
