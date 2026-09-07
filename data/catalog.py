"""
Multimodal E-Commerce Product Catalog
Rich heterogeneous product representations with multi-source evidence:
- Visual descriptors & reference images
- Tabular specifications
- Manufacturer marketing descriptions
- Multi-perspective customer reviews
- Ground-truth conflict definitions for evaluation
"""

from typing import List, Dict, Any

PRODUCTS_DATA: List[Dict[str, Any]] = [
    {
        "id": "prod_001",
        "title": "Nordic Haven Luxe Lounge Armchair",
        "category": "Living Room Furniture",
        "price": 24999,
        "currency": "INR",
        "brand": "Nordic Haven",
        "image_url": "https://images.unsplash.com/photo-1580481077195-731da03fed74?w=600&auto=format&fit=crop&q=80",
        "visual_features": {
            "primary_color": "Cognac Brown",
            "material_appearance": "Glossy textured leather with visible machine stitching",
            "style": "Mid-century Scandinavian",
            "dimensions_visual": "Deep seated armchair with angled solid wood tapered legs"
        },
        "specs": {
            "Upholstery Material": "Polyurethane (PU) Faux Leather",
            "Frame Material": "Engineered Pine Wood",
            "Leg Material": "Rubberwood with Walnut Stain",
            "Dimensions (WxDxH)": "82cm x 85cm x 88cm",
            "Weight Capacity": "120 kg",
            "Care Instructions": "Wipe clean with damp cloth only; avoid leather conditioners"
        },
        "manufacturer_description": (
            "Indulge in unparalleled Scandinavian luxury. Handcrafted with 100% genuine Italian full-grain leather, "
            "the Nordic Haven armchair brings timeless warmth and supple comfort to any contemporary living room. "
            "Our breathable leather will age gracefully, developing an exquisite patina over decades of use."
        ),
        "reviews": [
            {
                "reviewer": "Siddharth K. (Verified Buyer)",
                "rating": 2,
                "title": "Not real leather! Started peeling within a month",
                "text": "The chair looks attractive initially, but after 3 weeks of daily use, the surface at the armrest started peeling off in thin rubbery sheets. It is definitely synthetic PU leather, not genuine Italian full-grain as claimed in the title."
            },
            {
                "reviewer": "Pooja M. (Verified Buyer)",
                "rating": 3,
                "title": "Decent shape but mislabeled materials",
                "text": "Comfort is okay for the price, but be warned: this is plastic faux leather. Even the spec sheet inside the box says PU. Misleading description."
            },
            {
                "reviewer": "Arjun R.",
                "rating": 4,
                "title": "Stylish aesthetic",
                "text": "Comfortable seating angle and looks great in photos. Feels a bit warm after sitting for an hour."
            }
        ],
        "ground_truth_conflict": {
            "has_conflict": True,
            "conflict_type": "Material Authenticity Discrepancy",
            "severity": "HIGH",
            "discrepancy_details": (
                "Manufacturer description claims '100% genuine Italian full-grain leather', but the specification table "
                "explicitly states 'Polyurethane (PU) Faux Leather', and multiple verified buyer reviews confirm peeling."
            ),
            "calibrated_confidence": 0.38,
            "verdict": "CAUTION"
        }
    },
    {
        "id": "prod_002",
        "title": "ErgoElite Pro Ergonomic Executive Task Chair",
        "category": "Office Furniture",
        "price": 18500,
        "currency": "INR",
        "brand": "ErgoElite",
        "image_url": "https://images.unsplash.com/photo-1505797149-43b0069ec26b?w=600&auto=format&fit=crop&q=80",
        "visual_features": {
            "primary_color": "Stealth Black",
            "material_appearance": "High-tension breathable elastomeric mesh back with molded foam seat",
            "style": "Modern High-Performance Ergonomic",
            "dimensions_visual": "High-back chair with 3D adjustable armrests and 5-wheel star base"
        },
        "specs": {
            "Backrest Material": "Breathable Nylon Mesh",
            "Base Material": "Reinforced Glass-Filled Nylon",
            "Cylinder Class": "Class 3 Gas Lift",
            "Maximum Weight Capacity": "100 kg (220 lbs)",
            "Seat Height Adjustment": "45cm - 55cm",
            "Warranty": "3-Year Limited"
        },
        "manufacturer_description": (
            "Engineered for relentless productivity. The ErgoElite Pro features military-grade heavy-duty construction "
            "rated to comfortably support heavy-set users up to 160 kg (350 lbs) all day long. Equipped with dynamic lumbar support."
        ),
        "reviews": [
            {
                "reviewer": "Vikram N. (Verified Buyer)",
                "rating": 1,
                "title": "Dangerous! Cylinder sank under 115kg",
                "text": "I weigh 115 kg and bought this because the description proudly says rated up to 160 kg. Within two days, the gas lift kept dropping to the floor. Look at the sticker under the base: it clearly says MAX LOAD: 100 KG!"
            },
            {
                "reviewer": "Meera S. (Verified Buyer)",
                "rating": 4,
                "title": "Great for lighter individuals",
                "text": "I weigh 58 kg and it supports my lower back wonderfully. Lumbar curve is very responsive."
            }
        ],
        "ground_truth_conflict": {
            "has_conflict": True,
            "conflict_type": "Weight Capacity Mismatch",
            "severity": "CRITICAL",
            "discrepancy_details": (
                "Manufacturer description claims 'supports up to 160 kg (350 lbs)', whereas the manufacturer's own spec table "
                "specifies a 'Maximum Weight Capacity: 100 kg' Class 3 lift, supported by buyer failure reports."
            ),
            "calibrated_confidence": 0.32,
            "verdict": "AVOID"
        }
    },
    {
        "id": "prod_003",
        "title": "Aura Antique Brass Minimalist Pendant Light",
        "category": "Lighting & Decor",
        "price": 4200,
        "currency": "INR",
        "brand": "Aura Luminaires",
        "image_url": "https://images.unsplash.com/photo-1513506003901-1e6a229e2d15?w=600&auto=format&fit=crop&q=80",
        "visual_features": {
            "primary_color": "Warm Brass Gold & Matte Off-White",
            "material_appearance": "Metallic polished gold canopy with smooth matte ceramic dome",
            "style": "Nordic Contemporary",
            "dimensions_visual": "Compact suspended cone pendant"
        },
        "specs": {
            "Cap & Canopy Material": "Electroplated ABS Plastic (Brass Color)",
            "Shade Material": "Glazed Terracotta Ceramic",
            "Bulb Socket": "E27 (Max 40W)",
            "Cord Length": "1.2m adjustable braided fabric",
            "Weight": "1.1 kg"
        },
        "manufacturer_description": (
            "Elevate your kitchen island with genuine solid brass accents and artisan earthenware ceramic. "
            "Every metal component is forged from premium architectural brass designed to develop a rich, timeless patina."
        ),
        "reviews": [
            {
                "reviewer": "Karan D. (Verified Buyer)",
                "rating": 2,
                "title": "It is plastic, not brass!",
                "text": "Very disappointed. The description says solid brass accents, but the top cap is cheap lightweight plastic coated with shiny yellow paint. It even chipped while my electrician installed it."
            },
            {
                "reviewer": "Tara B.",
                "rating": 4,
                "title": "Looks good from afar",
                "text": "The ceramic shade itself is very nice and heavy. Just don't touch the top part because you can tell it's plastic."
            }
        ],
        "ground_truth_conflict": {
            "has_conflict": True,
            "conflict_type": "Material Composition & Finish Discrepancy",
            "severity": "MEDIUM",
            "discrepancy_details": (
                "Description states 'genuine solid brass forged metal', but the specification table discloses 'Electroplated ABS Plastic', "
                "and customer reviews corroborate the finish chipping easily."
            ),
            "calibrated_confidence": 0.52,
            "verdict": "CAUTION"
        }
    },
    {
        "id": "prod_004",
        "title": "Komorebi Solid Teakwood Coffee Table",
        "category": "Living Room Furniture",
        "price": 28900,
        "currency": "INR",
        "brand": "Komorebi Woodworks",
        "image_url": "https://images.unsplash.com/photo-1533090481720-856c6e3c1fdc?w=600&auto=format&fit=crop&q=80",
        "visual_features": {
            "primary_color": "Golden Teak & Natural Woodgrain",
            "material_appearance": "Subtle organic matte oil finish with natural knots and dovetail joinery",
            "style": "Japandi / Organic Modern",
            "dimensions_visual": "Low-profile rectangular table with tapered solid wood legs"
        },
        "specs": {
            "Material": "100% Solid Kiln-Dried Indonesian Teak (Grade A)",
            "Finish": "Zero-VOC Natural Linseed Oil",
            "Dimensions (WxDxH)": "110cm x 60cm x 42cm",
            "Weight": "19.2 kg",
            "Assembly Required": "Legs only (tools included)",
            "Certification": "FSC Certified Sustainably Harvested"
        },
        "manufacturer_description": (
            "Crafted from 100% solid Grade-A kiln-dried teak, the Komorebi table celebrates the organic beauty of genuine timber. "
            "Finished with non-toxic natural oils, each table features unique grain patterns with exceptional durability."
        ),
        "reviews": [
            {
                "reviewer": "Ramesh P. (Verified Buyer)",
                "rating": 5,
                "title": "Substantial, genuine solid timber",
                "text": "Had my master carpenter inspect it upon delivery—it is 100% solid genuine teak throughout, no veneers or MDF core. Extremely heavy and built to last generations."
            },
            {
                "reviewer": "Ananya V. (Verified Buyer)",
                "rating": 5,
                "title": "Breathtaking craftsmanship",
                "text": "Matches the photos exactly. The organic wood smell is soothing and the dovetail joints are flawless."
            }
        ],
        "ground_truth_conflict": {
            "has_conflict": False,
            "conflict_type": "None (Fully Grounded & Consistent)",
            "severity": "NONE",
            "discrepancy_details": (
                "All modalities agree: High-resolution visual grain matches genuine solid teak; spec table, description, "
                "and expert verified buyer reviews are completely harmonious with no contradictory claims."
            ),
            "calibrated_confidence": 0.95,
            "verdict": "RECOMMENDED"
        }
    },
    {
        "id": "prod_005",
        "title": "Zenith Whisper-Quiet Bladeless Aeroflow Fan",
        "category": "Home Appliances",
        "price": 12499,
        "currency": "INR",
        "brand": "Zenith Climate",
        "image_url": "https://images.unsplash.com/photo-1585771724684-38269d6639fd?w=600&auto=format&fit=crop&q=80",
        "visual_features": {
            "primary_color": "Matte White with Silver Trim",
            "material_appearance": "Sleek aerodynamic bladeless loop with touch LED display",
            "style": "Minimalist High-Tech",
            "dimensions_visual": "Slim vertical floor tower"
        },
        "specs": {
            "Motor Type": "Brushless DC Motor",
            "Airflow Speed": "8-Speed Digital Control",
            "Operational Noise Level": "46 dB (Low) to 62 dB (High)",
            "Power Consumption": "35W",
            "Dimensions": "22cm x 22cm x 95cm"
        },
        "manufacturer_description": (
            "Experience undisturbed tranquility. The Zenith Aeroflow features whisper-quiet acoustic dampening technology "
            "emitting less than 20 dB of sound, engineered specifically for peaceful bedrooms and nursery rooms."
        ),
        "reviews": [
            {
                "reviewer": "Deepak T. (Verified Buyer)",
                "rating": 2,
                "title": "Way too noisy for bedroom sleep",
                "text": "The marketing says under 20dB whisper quiet. I tested it with a sound level meter: even on speed 1 it registered 45dB, and on speed 5 it hits 59dB. It is louder than my regular ceiling fan."
            },
            {
                "reviewer": "Simran K.",
                "rating": 3,
                "title": "Looks great, but not silent",
                "text": "Modern aesthetic is gorgeous, but do not expect silent operation."
            }
        ],
        "ground_truth_conflict": {
            "has_conflict": True,
            "conflict_type": "Acoustic / Noise Rating Mismatch",
            "severity": "HIGH",
            "discrepancy_details": (
                "Marketing claims '<20 dB whisper quiet nursery grade', but the specification table lists '46 dB to 62 dB', "
                "and customer reviews independently measured up to 59dB noise levels."
            ),
            "calibrated_confidence": 0.40,
            "verdict": "CAUTION"
        }
    },
    {
        "id": "prod_006",
        "title": "StudioCraft Artisan Speckled Stoneware Dinnerware (16-Piece)",
        "category": "Dining & Kitchen",
        "price": 6800,
        "currency": "INR",
        "brand": "StudioCraft",
        "image_url": "https://images.unsplash.com/photo-1614707267537-b85aaf00c4b7?w=600&auto=format&fit=crop&q=80",
        "visual_features": {
            "primary_color": "Earthy Oatmeal Speckled Beige with Raw Rim",
            "material_appearance": "Semi-matte high-fired stoneware ceramic with exposed clay foot",
            "style": "Wabi-Sabi Rustic Artisan",
            "dimensions_visual": "Full 16-piece set of dinner plates, salad plates, soup bowls, and mugs"
        },
        "specs": {
            "Material": "High-Fired Dense Stoneware Clay",
            "Glaze": "Lead-Free & Cadmium-Free Food-Safe Matte Glaze",
            "Microwave Safe": "Yes",
            "Dishwasher Safe": "Yes",
            "Oven Safe": "Up to 220°C (428°F)",
            "Total Weight": "9.8 kg"
        },
        "manufacturer_description": (
            "Dine with artisanal warmth. StudioCraft stoneware is kiln-fired at 1280°C for exceptional chip resistance "
            "and daily durability. Fully dishwasher, microwave, and oven safe up to 220°C."
        ),
        "reviews": [
            {
                "reviewer": "Nidhi G. (Verified Buyer)",
                "rating": 5,
                "title": "Heavy, durable, and truly dishwasher proof",
                "text": "We have run these through our intensive dishwasher cycle for 4 months now—zero scratches, zero chips, and no cracking. Truly high-fired stoneware."
            },
            {
                "reviewer": "Raghav S. (Verified Buyer)",
                "rating": 5,
                "title": "Exactly as described",
                "text": "Beautiful organic weight and textured feel. Feels like pieces bought from a high-end pottery studio."
            }
        ],
        "ground_truth_conflict": {
            "has_conflict": False,
            "conflict_type": "None (Fully Grounded & Consistent)",
            "severity": "NONE",
            "discrepancy_details": (
                "Verified consistent evidence across visuals, heat/wash specifications, and verified buyer longevity reviews."
            ),
            "calibrated_confidence": 0.94,
            "verdict": "RECOMMENDED"
        }
    }
]

def get_product_by_id(product_id: str) -> Dict[str, Any]:
    for p in PRODUCTS_DATA:
        if p["id"] == product_id:
            return p
    return None
