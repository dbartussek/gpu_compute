use derivative::Derivative;
use itertools::Itertools;
use serde::{Deserialize, Deserializer, Serialize};
use smallstr::SmallString;
use std::{fs::File, io::BufReader};

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct Coord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Derivative)]
#[derivative(Default)]
#[derive(Serialize, Deserialize, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
#[repr(u8)]
pub enum Allegiance {
    #[derivative(Default)]
    Independent,

    Federation,
    Empire,
    Alliance,

    #[serde(rename = "Pilots Federation")]
    PilotsFederation,
}

#[derive(Derivative)]
#[derivative(Default)]
#[derive(Serialize, Deserialize, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
#[repr(u8)]
pub enum Government {
    #[derivative(Default)]
    Anarchy,

    Corporate,
    Confederacy,
    Democracy,
    Communism,
    Patronage,
    Dictatorship,
    Feudal,
    Cooperative,
    Theocracy,
    Prison,
    #[serde(rename = "Prison colony")]
    PrisonColony,
}


#[derive(Serialize, Deserialize, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
#[repr(u8)]
pub enum StationType {
    #[serde(rename = "Fleet Carrier")]
    FleetCarrier,

    #[serde(rename = "Mega ship")]
    MegaShip,

    #[serde(rename = "Odyssey Settlement")]
    OdysseySettlement,
    #[serde(rename = "Planetary Outpost")]
    PlanetaryOutpost,
    #[serde(rename = "Planetary Port")]
    PlanetaryPort,

    Outpost,

    #[serde(rename = "Coriolis Starport")]
    CoriolisStarport,
    #[serde(rename = "Ocellus Starport")]
    OcellusStarport,
    #[serde(rename = "Orbis Starport")]
    OrbisStarport,

    #[serde(rename = "Asteroid base")]
    AsteroidBase,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Station {
    pub id: u32,
    pub market_id: u64,

    #[serde(rename = "type")]
    pub ty: StationType,

    pub name: SmallString<[u8; 16]>,

    pub have_market: bool,
    pub have_shipyard: bool,
    pub have_outfitting: bool,
}


#[derive(Serialize, Deserialize, Clone)]
pub struct PopulatedSystem {
    #[serde(rename = "id64")]
    pub id: u64,

    pub name: SmallString<[u8; 16]>,
    pub coords: Coord,

    #[serde(deserialize_with = "option_to_default")]
    pub allegiance: Allegiance,
    #[serde(deserialize_with = "option_to_default")]
    pub government: Government,

    #[serde(deserialize_with = "option_to_default")]
    pub population: u64,

    pub stations: Vec<Station>,
}

fn option_to_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Deserialize::deserialize(d).map(|x: Option<_>| x.unwrap_or_default())
}

fn main() {
    let data: Vec<PopulatedSystem> = serde_json::from_reader(BufReader::new(
        File::open("datasets/systemsPopulated.json").unwrap(),
    ))
    .unwrap();

    println!("Loaded");

    let stations = data
        .iter()
        .map(|it| it.stations.clone())
        .flatten()
        .collect_vec();

    println!("{} Systems", data.len());
    println!("{} People", data.iter().map(|d| d.population).sum::<u64>());

    println!();
    println!("{} stations", stations.len());
    println!(
        "{} fleet carriers",
        stations
            .iter()
            .filter(|s| s.ty == StationType::FleetCarrier)
            .count()
    );
}
